"""
Knowledge distillation module for transferring knowledge from teacher to student model.
Implements various distillation techniques including response-based, feature-based, and logit-based approaches.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import time
from tqdm import tqdm

from src.models.teacher_model import TeacherLLM
from src.models.student_model import StudentLLM


class DistillationLoss:
    """
    Collection of loss functions for knowledge distillation.
    """
    
    @staticmethod
    def kl_divergence_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0,
        reduction: str = "batchmean"
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between student and teacher logits.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            temperature: Temperature for softening probability distributions
            reduction: Reduction method for the loss
            
        Returns:
            KL divergence loss
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Shape mismatch: student_logits {student_logits.shape} vs "
                f"teacher_logits {teacher_logits.shape}"
            )
            
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        
        # Apply softmax to convert logits to probabilities
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence
        loss = F.kl_div(
            student_probs.log(),
            teacher_probs,
            reduction=reduction
        )
        
        # Scale by temperature squared as in the original paper
        return loss * (temperature ** 2)
    
    @staticmethod
    def mse_loss(
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute MSE loss between student and teacher outputs.
        
        Args:
            student_outputs: Outputs from student model (logits or hidden states)
            teacher_outputs: Outputs from teacher model (logits or hidden states)
            reduction: Reduction method for the loss
            
        Returns:
            MSE loss
        """
        return F.mse_loss(student_outputs, teacher_outputs, reduction=reduction)
    
    @staticmethod
    def cosine_similarity_loss(
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss between student and teacher outputs.
        
        Args:
            student_outputs: Outputs from student model (logits or hidden states)
            teacher_outputs: Outputs from teacher model (logits or hidden states)
            reduction: Reduction method for the loss
            
        Returns:
            Cosine similarity loss (1 - cosine_similarity)
        """
        # Normalize the vectors
        student_norm = F.normalize(student_outputs, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_outputs, p=2, dim=-1)
        
        # Compute cosine similarity
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=-1)
        
        # Convert to loss (1 - similarity)
        loss = 1.0 - cosine_sim
        
        # Apply reduction
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss


class DistillationDataset(Dataset):
    """
    Dataset for knowledge distillation.
    """
    
    def __init__(
        self,
        texts: List[str],
        teacher_tokenizer,
        student_tokenizer,
        max_length: int = 512,
        teacher_outputs: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples for distillation
            teacher_tokenizer: Tokenizer for the teacher model
            student_tokenizer: Tokenizer for the student model
            max_length: Maximum sequence length
            teacher_outputs: Pre-computed teacher outputs (optional)
        """
        self.texts = texts
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_length = max_length
        self.teacher_outputs = teacher_outputs
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize for teacher and student
        teacher_encoding = self.teacher_tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        student_encoding = self.student_tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        teacher_encoding = {k: v.squeeze(0) for k, v in teacher_encoding.items()}
        student_encoding = {k: v.squeeze(0) for k, v in student_encoding.items()}
        
        # Add pre-computed teacher outputs if available
        if self.teacher_outputs is not None and idx in self.teacher_outputs:
            teacher_encoding.update(self.teacher_outputs[idx])
            
        return {
            "teacher": teacher_encoding,
            "student": student_encoding,
            "text": text
        }


class KnowledgeDistillation:
    """
    Knowledge distillation pipeline for transferring knowledge from teacher to student model.
    """
    
    def __init__(
        self,
        teacher_model: TeacherLLM,
        student_model: StudentLLM,
        distillation_type: str = "response",  # "response", "feature", "logit", or "combined"
        temperature: float = 2.0,
        alpha: float = 0.5,  # Weight for distillation loss vs task loss
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "checkpoints/distillation",
        **kwargs
    ):
        """
        Initialize the knowledge distillation pipeline.
        
        Args:
            teacher_model: Teacher LLM instance
            student_model: Student LLM instance
            distillation_type: Type of distillation to perform
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss vs task loss
            device: Device to perform distillation on
            output_dir: Directory to save checkpoints
            **kwargs: Additional arguments
        """
        self.teacher = teacher_model
        self.student = student_model
        self.distillation_type = distillation_type
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "distillation.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("KnowledgeDistillation")
        
        # Ensure both models are on the same device
        if self.teacher.device != device:
            self.logger.warning(
                f"Teacher model is on {self.teacher.device}, moving to {device}"
            )
            self.teacher.model.to(device)
            self.teacher.device = device
            
        if self.student.device != device:
            self.logger.warning(
                f"Student model is on {self.student.device}, moving to {device}"
            )
            self.student.model.to(device)
            self.student.device = device
            
        self.logger.info(f"Initialized knowledge distillation with type: {distillation_type}")
        self.logger.info(f"Temperature: {temperature}, Alpha: {alpha}")
        
    def prepare_dataset(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        num_workers: int = 4,
        precompute_teacher: bool = True
    ) -> Tuple[DataLoader, Optional[Dict]]:
        """
        Prepare dataset for distillation.
        
        Args:
            texts: List of text samples for distillation
            batch_size: Batch size for data loader
            max_length: Maximum sequence length
            num_workers: Number of workers for data loader
            precompute_teacher: Whether to precompute teacher outputs
            
        Returns:
            DataLoader and optionally precomputed teacher outputs
        """
        self.logger.info(f"Preparing dataset with {len(texts)} samples")
        
        # Precompute teacher outputs if requested
        teacher_outputs = None
        if precompute_teacher and self.distillation_type in ["logit", "feature", "combined"]:
            self.logger.info("Precomputing teacher outputs...")
            teacher_outputs = {}
            
            # Process in batches to avoid OOM
            batch_size_precompute = 16
            for i in tqdm(range(0, len(texts), batch_size_precompute)):
                batch_texts = texts[i:i+batch_size_precompute]
                batch_outputs = self.teacher.get_hidden_states(
                    batch_texts,
                    output_hidden_states=True
                )
                
                # Store outputs for each sample
                for j, text in enumerate(batch_texts):
                    idx = i + j
                    teacher_outputs[idx] = {
                        "teacher_logits": batch_outputs["logits"][j].detach(),
                        "teacher_hidden_states": [
                            h[j].detach() for h in batch_outputs["hidden_states"]
                        ] if batch_outputs["hidden_states"] else None
                    }
            
            self.logger.info(f"Precomputed teacher outputs for {len(teacher_outputs)} samples")
        
        # Create dataset
        dataset = DistillationDataset(
            texts=texts,
            teacher_tokenizer=self.teacher.tokenizer,
            student_tokenizer=self.student.tokenizer,
            max_length=max_length,
            teacher_outputs=teacher_outputs
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader, teacher_outputs
    
    def compute_distillation_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute distillation loss based on the specified type.
        
        Args:
            student_outputs: Outputs from student model
            teacher_outputs: Outputs from teacher model
            
        Returns:
            Distillation loss
        """
        if self.distillation_type == "response":
            # Response-based distillation (final layer outputs)
            return DistillationLoss.kl_divergence_loss(
                student_outputs["logits"],
                teacher_outputs["logits"],
                temperature=self.temperature
            )
        
        elif self.distillation_type == "feature":
            # Feature-based distillation (intermediate representations)
            student_hidden = student_outputs["hidden_states"][-1]  # Last hidden layer
            teacher_hidden = teacher_outputs["hidden_states"][-1]  # Last hidden layer
            
            # If dimensions don't match, project student hidden states
            if student_hidden.shape != teacher_hidden.shape:
                # Simple linear projection (in practice, you might want a more sophisticated approach)
                projection = nn.Linear(
                    student_hidden.shape[-1],
                    teacher_hidden.shape[-1],
                    device=self.device
                )
                student_hidden = projection(student_hidden)
                
            return DistillationLoss.mse_loss(student_hidden, teacher_hidden)
        
        elif self.distillation_type == "logit":
            # Logit-based distillation
            return DistillationLoss.kl_divergence_loss(
                student_outputs["logits"],
                teacher_outputs["logits"],
                temperature=self.temperature
            )
        
        elif self.distillation_type == "combined":
            # Combined distillation (logit + feature)
            logit_loss = DistillationLoss.kl_divergence_loss(
                student_outputs["logits"],
                teacher_outputs["logits"],
                temperature=self.temperature
            )
            
            # Feature loss from last hidden layer
            student_hidden = student_outputs["hidden_states"][-1]
            teacher_hidden = teacher_outputs["hidden_states"][-1]
            
            # If dimensions don't match, project student hidden states
            if student_hidden.shape != teacher_hidden.shape:
                projection = nn.Linear(
                    student_hidden.shape[-1],
                    teacher_hidden.shape[-1],
                    device=self.device
                )
                student_hidden = projection(student_hidden)
                
            feature_loss = DistillationLoss.cosine_similarity_loss(
                student_hidden, teacher_hidden
            )
            
            # Combine losses with equal weight (can be parameterized)
            return 0.5 * logit_loss + 0.5 * feature_loss
        
        else:
            raise ValueError(f"Unknown distillation type: {self.distillation_type}")
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: Optional[int] = None,
        scheduler_type: str = "linear",
        fp16: bool = False,
        callback: Optional[Callable] = None
    ):
        """
        Train the student model using knowledge distillation.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for gradient clipping
            logging_steps: Number of steps between logging
            save_steps: Number of steps between saving checkpoints
            eval_steps: Number of steps b
(Content truncated due to size limit. Use line ranges to read in chunks)