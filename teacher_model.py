"""
Teacher model implementation for knowledge distillation.
This module provides a wrapper around a large language model to serve as the teacher.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple, Union


class TeacherLLM:
    """
    Teacher LLM class that wraps a pre-trained large language model.
    This serves as the knowledge source for distillation to the student model.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "fp16",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the teacher model.
        
        Args:
            model_name_or_path: HuggingFace model name or path to local model
            device: Device to load the model on ('cuda', 'cpu', etc.)
            precision: Model precision ('fp16', 'fp32', 'bf16')
            cache_dir: Directory to cache the downloaded model
            **kwargs: Additional arguments to pass to the model loading function
        """
        self.model_name = model_name_or_path
        self.device = device
        self.precision = precision
        self.cache_dir = cache_dir
        
        # Set up dtype based on precision
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
            
        print(f"Loading teacher model: {model_name_or_path}")
        print(f"Device: {device}, Precision: {precision}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            padding_side="left",
            **kwargs
        )
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model configuration
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **kwargs
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=self.config,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            device_map=device if device == "auto" else None,
            **kwargs
        )
        
        if device != "auto":
            self.model.to(device)
            
        # Set evaluation mode
        self.model.eval()
        
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text based on prompts.
        
        Args:
            prompts: Input text prompt(s)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional arguments to pass to the generate method
            
        Returns:
            List of generated text responses
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        # Extract only the newly generated text (remove the prompt)
        results = []
        for i, text in enumerate(generated_texts):
            if text.startswith(prompts[i]):
                results.append(text[len(prompts[i]):].strip())
            else:
                results.append(text.strip())
                
        return results
    
    def get_hidden_states(
        self,
        prompts: Union[str, List[str]],
        output_hidden_states: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Get hidden states from the model for knowledge distillation.
        
        Args:
            prompts: Input text prompt(s)
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Dictionary containing hidden states and other model outputs
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model outputs with hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                **kwargs
            )
            
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
            "logits": outputs.logits,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask
        }
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model and tokenizer to a directory.
        
        Args:
            save_directory: Directory to save the model to
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}")
