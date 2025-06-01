"""
Quantization module for model compression.
Provides utilities for post-training quantization and quantization-aware training.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """Configuration for quantization methods."""
    
    # Quantization bit-width
    bits: int = 8  # 8, 4, or 2 bits
    
    # Quantization scheme
    scheme: str = "symmetric"  # "symmetric" or "asymmetric"
    
    # Quantization granularity
    granularity: str = "per_tensor"  # "per_tensor" or "per_channel"
    
    # Quantization type
    quantization_type: str = "static"  # "static" or "dynamic"
    
    # Whether to quantize activations
    quantize_activations: bool = False
    
    # Whether to use mixed precision (keep some layers in higher precision)
    mixed_precision: bool = True
    
    # Layers to exclude from quantization
    excluded_layers: List[str] = None


class Quantizer:
    """
    Model quantization for compression and inference acceleration.
    """
    
    def __init__(
        self,
        config: Optional[QuantizationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the quantizer.
        
        Args:
            config: Quantization configuration
            logger: Logger instance
        """
        self.config = config or QuantizationConfig()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Initialized quantizer with config: {self.config}")
        
    def quantize_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        calibration_dataset: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> AutoModelForCausalLM:
        """
        Quantize a model using static post-training quantization.
        
        Args:
            model: Model to quantize
            tokenizer: Tokenizer for the model
            calibration_dataset: Optional dataset for calibration
            output_dir: Optional directory to save the quantized model
            
        Returns:
            Quantized model
        """
        self.logger.info(f"Quantizing model to {self.config.bits} bits")
        
        # Set model to evaluation mode
        model.eval()
        
        # Different quantization approaches based on bit-width
        if self.config.bits == 8:
            # 8-bit quantization using bitsandbytes
            try:
                import bitsandbytes as bnb
                
                # Convert linear layers to 8-bit
                model = self._convert_to_8bit(model)
                
                self.logger.info("Model converted to 8-bit precision")
                
            except ImportError:
                self.logger.error("bitsandbytes not installed. Please install it for 8-bit quantization.")
                raise
                
        elif self.config.bits == 4:
            # 4-bit quantization using bitsandbytes or optimum
            try:
                import bitsandbytes as bnb
                
                # Convert linear layers to 4-bit
                model = self._convert_to_4bit(model)
                
                self.logger.info("Model converted to 4-bit precision")
                
            except ImportError:
                self.logger.error("bitsandbytes not installed. Please install it for 4-bit quantization.")
                raise
                
        elif self.config.bits == 2:
            # 2-bit quantization (experimental)
            self.logger.warning("2-bit quantization is experimental and may result in significant accuracy loss")
            
            # Use custom 2-bit quantization
            model = self._custom_quantize(model, bits=2)
            
        else:
            raise ValueError(f"Unsupported bit-width: {self.config.bits}")
            
        # Save quantized model if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save quantization config
            import json
            with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
                json.dump({
                    "bits": self.config.bits,
                    "scheme": self.config.scheme,
                    "granularity": self.config.granularity,
                    "quantization_type": self.config.quantization_type,
                    "quantize_activations": self.config.quantize_activations,
                    "mixed_precision": self.config.mixed_precision,
                    "excluded_layers": self.config.excluded_layers
                }, f, indent=2)
                
            self.logger.info(f"Saved quantized model to {output_dir}")
            
        return model
    
    def _convert_to_8bit(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Convert model to 8-bit precision using bitsandbytes.
        
        Args:
            model: Model to convert
            
        Returns:
            8-bit quantized model
        """
        import bitsandbytes as bnb
        
        # Get list of linear layer names to exclude
        excluded_modules = set()
        if self.config.excluded_layers:
            for name in self.config.excluded_layers:
                excluded_modules.add(name)
                
        # Replace linear layers with 8-bit equivalents
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in excluded_modules:
                # Skip if this is an output layer and we're using mixed precision
                if self.config.mixed_precision and any(n in name for n in ["output", "classifier", "lm_head"]):
                    self.logger.info(f"Keeping {name} in full precision (mixed precision)")
                    continue
                    
                # Replace with 8-bit linear
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                
                # Create 8-bit linear layer
                eight_bit_linear = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0
                )
                
                # Copy weights and bias
                eight_bit_linear.weight.data = module.weight.data
                if module.bias is not None:
                    eight_bit_linear.bias.data = module.bias.data
                    
                # Replace module
                setattr(parent, child_name, eight_bit_linear)
                
        return model
    
    def _convert_to_4bit(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Convert model to 4-bit precision using bitsandbytes.
        
        Args:
            model: Model to convert
            
        Returns:
            4-bit quantized model
        """
        import bitsandbytes as bnb
        
        # Get list of linear layer names to exclude
        excluded_modules = set()
        if self.config.excluded_layers:
            for name in self.config.excluded_layers:
                excluded_modules.add(name)
                
        # Determine quantization type
        if self.config.quantization_type == "nf4":
            compute_dtype = torch.float16
            quant_type = bnb.nn.modules.Linear4bit.ConfigNF4()
        else:  # fp4
            compute_dtype = torch.float16
            quant_type = bnb.nn.modules.Linear4bit.ConfigFP4()
                
        # Replace linear layers with 4-bit equivalents
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in excluded_modules:
                # Skip if this is an output layer and we're using mixed precision
                if self.config.mixed_precision and any(n in name for n in ["output", "classifier", "lm_head"]):
                    self.logger.info(f"Keeping {name} in full precision (mixed precision)")
                    continue
                    
                # Replace with 4-bit linear
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                
                # Create 4-bit linear layer
                four_bit_linear = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type
                )
                
                # Copy weights and bias
                # Note: This is a simplified approach; in practice, proper 4-bit conversion
                # requires more sophisticated handling
                four_bit_linear.weight.data = module.weight.data
                if module.bias is not None and hasattr(four_bit_linear, "bias"):
                    four_bit_linear.bias.data = module.bias.data
                    
                # Replace module
                setattr(parent, child_name, four_bit_linear)
                
        return model
    
    def _custom_quantize(self, model: AutoModelForCausalLM, bits: int = 2) -> AutoModelForCausalLM:
        """
        Apply custom low-bit quantization to model.
        
        Args:
            model: Model to quantize
            bits: Number of bits for quantization
            
        Returns:
            Quantized model
        """
        self.logger.warning(f"Using experimental {bits}-bit quantization")
        
        # Get list of linear layer names to exclude
        excluded_modules = set()
        if self.config.excluded_layers:
            for name in self.config.excluded_layers:
                excluded_modules.add(name)
                
        # Custom quantization function
        def quantize_tensor(tensor, num_bits, scheme="symmetric"):
            # Determine range
            if scheme == "symmetric":
                max_val = torch.max(torch.abs(tensor))
                min_val = -max_val
            else:  # asymmetric
                max_val = torch.max(tensor)
                min_val = torch.min(tensor)
                
            # Compute scale and zero point
            q_max = 2 ** num_bits - 1
            scale = (max_val - min_val) / q_max
            zero_point = -min_val / scale
            
            # Quantize
            tensor_q = torch.round(tensor / scale + zero_point)
            tensor_q = torch.clamp(tensor_q, 0, q_max)
            
            # Dequantize (for simulation)
            tensor_dq = (tensor_q - zero_point) * scale
            
            return tensor_dq, scale, zero_point
            
        # Apply quantization to model parameters
        for name, param in model.named_parameters():
            # Skip excluded layers
            if any(excluded in name for excluded in excluded_modules):
                continue
                
            # Skip if this is an output layer and we're using mixed precision
            if self.config.mixed_precision and any(n in name for n in ["output", "classifier", "lm_head"]):
                self.logger.info(f"Keeping {name} in full precision (mixed precision)")
                continue
                
            # Quantize parameter
            param_q, scale, zero_point = quantize_tensor(
                param.data,
                bits,
                scheme=self.config.scheme
            )
            
            # Replace parameter with quantized version
            param.data = param_q
            
            # Store quantization parameters (for reference)
            param._scale = scale
            param._zero_point = zero_point
            
        return model
    
    @staticmethod
    def load_quantized_model(
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a quantized model.
        
        Args:
            model_path: Path to the quantized model
            device: Device to load the model on
            **kwargs: Additional arguments to pass to the model loading function
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if quantization config exists
        config_path = os.path.join(model_path, "quantization_config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                quant_config = json.load(f)
                
            bits = quant_config.get("bits", 8)
        else:
            # Default to 8-bit if no config is found
            bits = 8
            
        # Load model based on quantization bit-width
        if bits == 8:
            try:
                import bitsandbytes as bnb
                
                # Load 8-bit model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_8bit=True,
                    device_map=device if device == "auto" else {"": device},
                    **kwargs
                )
                
            except ImportError:
                logging.warning("bitsandbytes not installed. Loading model in full precision.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device if device == "auto" else {"": device},
                    **kwargs
                )
                
        elif bits == 4:
            try:
                import bitsandbytes as bnb
                
                # Load 4-bit model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_4bit=True,
                    device_map=device if device == "auto" else {"": device},
                    **kwargs
                )
                
            except ImportError:
                logging.warning("bitsandbytes not installed. Loading model in full precision.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device if device == "auto" else {"": device},
                    **kwargs
                )
                
        else:
            # Load in full precision for other bit-widths
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device if device == "auto" else {"": device},
                **kwargs
            )
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        
        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
