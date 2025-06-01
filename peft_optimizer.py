"""
PEFT (Parameter-Efficient Fine-Tuning) implementation with QLoRA.
Provides utilities for efficient model adaptation with minimal memory footprint.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
import bitsandbytes as bnb
from dataclasses import dataclass


@dataclass
class PeftConfig:
    """Configuration for PEFT methods."""
    
    # LoRA parameters
    lora_r: int = 8  # Rank of LoRA matrices
    lora_alpha: int = 16  # Alpha parameter for LoRA scaling
    lora_dropout: float = 0.05  # Dropout probability for LoRA layers
    
    # Target modules to apply LoRA
    target_modules: Optional[List[str]] = None  # If None, will be auto-detected
    
    # QLoRA parameters
    quantization_bits: int = 4  # Quantization bit-width (4 or 8)
    quantization_type: str = "nf4"  # Quantization type (nf4, fp4, int8)
    double_quantization: bool = True  # Whether to use double quantization
    
    # Training parameters
    use_gradient_checkpointing: bool = True  # Whether to use gradient checkpointing
    
    # Modules to not apply LoRA
    modules_to_save: Optional[List[str]] = None  # Modules to save fully (not apply LoRA)


class PeftOptimizer:
    """
    Parameter-Efficient Fine-Tuning optimizer with QLoRA support.
    """
    
    def __init__(
        self,
        config: Optional[PeftConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the PEFT optimizer.
        
        Args:
            config: PEFT configuration
            logger: Logger instance
        """
        self.config = config or PeftConfig()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Initialized PEFT optimizer with config: {self.config}")
        
    def prepare_model_for_qlora(
        self,
        model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Prepare a model for QLoRA fine-tuning.
        
        Args:
            model_name_or_path: HuggingFace model name or path to local model
            device: Device to load the model on
            cache_dir: Directory to cache the model
            **kwargs: Additional arguments to pass to the model loading function
            
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Preparing model {model_name_or_path} for QLoRA")
        
        # Determine quantization type
        if self.config.quantization_bits == 4:
            if self.config.quantization_type == "nf4":
                quantization_config = bnb.nn.modules.Linear4bit.ConfigNF4()
            else:  # fp4
                quantization_config = bnb.nn.modules.Linear4bit.ConfigFP4()
                
            # Load model in 4-bit precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device if device == "auto" else {"": device},
                load_in_4bit=True,
                quantization_config=quantization_config,
                use_cache=not self.config.use_gradient_checkpointing,
                cache_dir=cache_dir,
                **kwargs
            )
            
        elif self.config.quantization_bits == 8:
            # Load model in 8-bit precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device if device == "auto" else {"": device},
                load_in_8bit=True,
                use_cache=not self.config.use_gradient_checkpointing,
                cache_dir=cache_dir,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unsupported quantization bits: {self.config.quantization_bits}")
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            padding_side="left",
            **kwargs
        )
        
        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Prepare model for kbit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing
        )
        
        # Auto-detect target modules if not specified
        target_modules = self.config.target_modules
        if target_modules is None:
            # Common module names for different model architectures
            target_modules = self._detect_target_modules(model)
            
        self.logger.info(f"Using target modules: {target_modules}")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=self.config.modules_to_save
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        self._print_trainable_parameters(model)
        
        return model, tokenizer
    
    def _detect_target_modules(self, model: AutoModelForCausalLM) -> List[str]:
        """
        Auto-detect target modules for LoRA.
        
        Args:
            model: Model to detect target modules for
            
        Returns:
            List of target module names
        """
        # Get model architecture
        architecture = model.config.architectures[0] if hasattr(model.config, "architectures") else "Unknown"
        
        # Common module patterns for different architectures
        if "Llama" in architecture:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "GPT2" in architecture:
            return ["c_attn", "c_proj", "c_fc"]
        elif "GPTNeo" in architecture or "GPTNeoX" in architecture:
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"]
        elif "OPT" in architecture:
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        elif "RoBERTa" in architecture or "BERT" in architecture:
            return ["query", "key", "value", "output.dense", "intermediate.dense"]
        else:
            # Generic approach: find all linear layers
            target_modules = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Extract the last part of the name (after the last dot)
                    parts = name.split(".")
                    if len(parts) > 0:
                        target_modules.append(parts[-1])
                        
            # Remove duplicates and sort
            target_modules = sorted(list(set(target_modules)))
            
            self.logger.warning(
                f"Auto-detected target modules for unknown architecture {architecture}: {target_modules}"
            )
            
            return target_modules
    
    def _print_trainable_parameters(self, model: PeftModel):
        """
        Print the number of trainable parameters in the model.
        
        Args:
            model: PEFT model
        """
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        self.logger.info(
            f"Trainable parameters: {trainable_params:,d} ({trainable_params / all_params:.2%})"
        )
        self.logger.info(f"All parameters: {all_params:,d}")
        
    def save_peft_model(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        output_dir: str
    ):
        """
        Save a PEFT model and tokenizer.
        
        Args:
            model: PEFT model
            tokenizer: Tokenizer
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PEFT model
        model.save_pretrained(output_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Saved PEFT model and tokenizer to {output_dir}")
        
    def load_peft_model(
        self,
        base_model_name_or_path: str,
        peft_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """
        Load a PEFT model.
        
        Args:
            base_model_name_or_path: Base model name or path
            peft_model_path: Path to PEFT model
            device: Device to load the model on
            **kwargs: Additional arguments to pass to the model loading function
            
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Loading PEFT model from {peft_model_path}")
        
        # Load base model with same quantization as during training
        if self.config.quantization_bits == 4:
            if self.config.quantization_type == "nf4":
                quantization_config = bnb.nn.modules.Linear4bit.ConfigNF4()
            else:  # fp4
                quantization_config = bnb.nn.modules.Linear4bit.ConfigFP4()
                
            # Load model in 4-bit precision
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                device_map=device if device == "auto" else {"": device},
                load_in_4bit=True,
                quantization_config=quantization_config,
                **kwargs
            )
            
        elif self.config.quantization_bits == 8:
            # Load model in 8-bit precision
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                device_map=device if device == "auto" else {"": device},
                load_in_8bit=True,
                **kwargs
            )
            
        else:
            # Load in full precision
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                device_map=device if device == "auto" else {"": device},
                **kwargs
            )
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            peft_model_path,
            **kwargs
        )
        
        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load PEFT model
        model = PeftModel.from_pretrained(
            model,
            peft_model_path,
            **kwargs
        )
        
        self.logger.info(f"Loaded PEFT model from {peft_model_path}")
        
        return model, tokenizer
        
    def merge_and_save(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        output_dir: str
    ):
        """
        Merge LoRA weights with base model and save.
        
        Args:
            model: PEFT model
            tokenizer: Tokenizer
            output_dir: Output directory
        """
        self.logger.info("Merging LoRA weights with base model")
        
        # Merge weights
        model = model.merge_and_unload()
        
        # Save merged model
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Saved merged model to {output_dir}")
        
        return model
