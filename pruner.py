"""
Pruning module for model compression.
Provides utilities for weight pruning and structured pruning.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dataclasses import dataclass


@dataclass
class PruningConfig:
    """Configuration for pruning methods."""
    
    # Pruning method
    method: str = "magnitude"  # "magnitude", "structured", "movement"
    
    # Pruning sparsity (percentage of weights to prune)
    sparsity: float = 0.3  # 0.0 to 1.0
    
    # Pruning schedule
    schedule: str = "one_shot"  # "one_shot", "iterative", "gradual"
    
    # Number of pruning iterations (for iterative pruning)
    n_iterations: int = 10
    
    # Pruning granularity
    granularity: str = "element"  # "element", "vector", "block"
    
    # Block size for block pruning
    block_size: Tuple[int, int] = (4, 4)
    
    # Layers to exclude from pruning
    excluded_layers: List[str] = None


class Pruner:
    """
    Model pruning for compression and inference acceleration.
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the pruner.
        
        Args:
            config: Pruning configuration
            logger: Logger instance
        """
        self.config = config or PruningConfig()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Initialized pruner with config: {self.config}")
        
    def prune_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        output_dir: Optional[str] = None
    ) -> AutoModelForCausalLM:
        """
        Prune a model using the specified method.
        
        Args:
            model: Model to prune
            tokenizer: Tokenizer for the model
            output_dir: Optional directory to save the pruned model
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Pruning model with method: {self.config.method}, sparsity: {self.config.sparsity}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Get list of prunable parameters
        prunable_params = self._get_prunable_parameters(model)
        
        # Apply pruning based on method
        if self.config.method == "magnitude":
            self._apply_magnitude_pruning(model, prunable_params)
        elif self.config.method == "structured":
            self._apply_structured_pruning(model, prunable_params)
        elif self.config.method == "movement":
            self.logger.warning("Movement pruning requires fine-tuning and is not implemented in this function")
            return model
        else:
            raise ValueError(f"Unsupported pruning method: {self.config.method}")
            
        # Save pruned model if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save pruning config
            import json
            with open(os.path.join(output_dir, "pruning_config.json"), "w") as f:
                json.dump({
                    "method": self.config.method,
                    "sparsity": self.config.sparsity,
                    "schedule": self.config.schedule,
                    "n_iterations": self.config.n_iterations,
                    "granularity": self.config.granularity,
                    "block_size": self.config.block_size,
                    "excluded_layers": self.config.excluded_layers
                }, f, indent=2)
                
            self.logger.info(f"Saved pruned model to {output_dir}")
            
        return model
    
    def _get_prunable_parameters(self, model: AutoModelForCausalLM) -> Dict[str, torch.nn.Parameter]:
        """
        Get prunable parameters from the model.
        
        Args:
            model: Model to get parameters from
            
        Returns:
            Dictionary of prunable parameters
        """
        prunable_params = {}
        
        # Get list of layer names to exclude
        excluded_modules = set()
        if self.config.excluded_layers:
            for name in self.config.excluded_layers:
                excluded_modules.add(name)
                
        # Collect prunable parameters
        for name, param in model.named_parameters():
            # Skip non-weight parameters (e.g., biases)
            if "weight" not in name:
                continue
                
            # Skip excluded layers
            if any(excluded in name for excluded in excluded_modules):
                continue
                
            # Skip output layers (important for model functionality)
            if any(n in name for n in ["output", "classifier", "lm_head"]):
                continue
                
            # Skip small parameters
            if param.numel() < 100:
                continue
                
            # Add to prunable parameters
            prunable_params[name] = param
            
        self.logger.info(f"Found {len(prunable_params)} prunable parameters")
        
        return prunable_params
    
    def _apply_magnitude_pruning(
        self,
        model: AutoModelForCausalLM,
        prunable_params: Dict[str, torch.nn.Parameter]
    ):
        """
        Apply magnitude-based pruning to the model.
        
        Args:
            model: Model to prune
            prunable_params: Dictionary of prunable parameters
        """
        # One-shot pruning
        if self.config.schedule == "one_shot":
            for name, param in prunable_params.items():
                # Create pruning mask
                mask = self._create_magnitude_mask(param, self.config.sparsity)
                
                # Apply mask
                param.data = param.data * mask
                
                # Store mask as attribute for future reference
                param._prune_mask = mask
                
            self.logger.info(f"Applied one-shot magnitude pruning with sparsity {self.config.sparsity}")
            
        # Iterative pruning
        elif self.config.schedule == "iterative":
            # Calculate per-iteration sparsity
            sparsity_per_iter = 1 - (1 - self.config.sparsity) ** (1 / self.config.n_iterations)
            
            for i in range(self.config.n_iterations):
                for name, param in prunable_params.items():
                    # Create pruning mask
                    mask = self._create_magnitude_mask(param, sparsity_per_iter)
                    
                    # Apply mask
                    param.data = param.data * mask
                    
                    # Store mask as attribute for future reference
                    if hasattr(param, "_prune_mask"):
                        param._prune_mask = param._prune_mask * mask
                    else:
                        param._prune_mask = mask
                        
                self.logger.info(f"Completed pruning iteration {i+1}/{self.config.n_iterations}")
                
            self.logger.info(f"Applied iterative magnitude pruning with final sparsity {self.config.sparsity}")
            
        # Gradual pruning
        elif self.config.schedule == "gradual":
            # Not implemented in this version
            self.logger.warning("Gradual pruning schedule not implemented, falling back to one-shot")
            
            for name, param in prunable_params.items():
                # Create pruning mask
                mask = self._create_magnitude_mask(param, self.config.sparsity)
                
                # Apply mask
                param.data = param.data * mask
                
                # Store mask as attribute for future reference
                param._prune_mask = mask
                
        else:
            raise ValueError(f"Unsupported pruning schedule: {self.config.schedule}")
    
    def _create_magnitude_mask(self, param: torch.nn.Parameter, sparsity: float) -> torch.Tensor:
        """
        Create a pruning mask based on weight magnitudes.
        
        Args:
            param: Parameter to create mask for
            sparsity: Pruning sparsity
            
        Returns:
            Binary mask tensor
        """
        # Element-wise pruning
        if self.config.granularity == "element":
            # Calculate threshold
            threshold = torch.quantile(torch.abs(param.data.flatten()), sparsity)
            
            # Create mask
            mask = (torch.abs(param.data) > threshold).float()
            
        # Vector-wise pruning (prune entire vectors)
        elif self.config.granularity == "vector":
            # For 2D weights, we can prune rows or columns
            if param.dim() == 2:
                # Calculate vector norms (along dim 1, i.e., rows)
                vector_norms = torch.norm(param.data, dim=1)
                
                # Calculate threshold
                threshold = torch.quantile(vector_norms, sparsity)
                
                # Create mask
                vector_mask = (vector_norms > threshold).float()
                
                # Expand mask to match parameter shape
                mask = vector_mask.unsqueeze(1).expand_as(param.data)
                
            else:
                # Fall back to element-wise for non-2D tensors
                threshold = torch.quantile(torch.abs(param.data.flatten()), sparsity)
                mask = (torch.abs(param.data) > threshold).float()
                
        # Block-wise pruning
        elif self.config.granularity == "block":
            # Only applicable to 2D weights
            if param.dim() == 2:
                # Get block size
                block_h, block_w = self.config.block_size
                
                # Reshape tensor into blocks
                h, w = param.shape
                h_blocks = h // block_h
                w_blocks = w // block_w
                
                # Pad if necessary
                h_pad = h_blocks * block_h
                w_pad = w_blocks * block_w
                
                if h_pad < h or w_pad < w:
                    # Create padded tensor
                    padded = torch.zeros((h_pad, w_pad), device=param.device)
                    padded[:h, :w] = param.data[:h, :w]
                    
                    # Calculate block norms
                    blocks = padded.reshape(h_blocks, block_h, w_blocks, block_w)
                    block_norms = torch.norm(blocks, dim=(1, 3))
                    
                    # Calculate threshold
                    threshold = torch.quantile(block_norms.flatten(), sparsity)
                    
                    # Create block mask
                    block_mask = (block_norms > threshold).float()
                    
                    # Expand mask to match block size
                    mask = block_mask.unsqueeze(1).unsqueeze(3).expand(h_blocks, block_h, w_blocks, block_w)
                    
                    # Reshape mask to match parameter shape
                    mask = mask.reshape(h_pad, w_pad)
                    
                    # Crop mask to original size
                    mask = mask[:h, :w]
                    
                else:
                    # Calculate block norms
                    blocks = param.data.reshape(h_blocks, block_h, w_blocks, block_w)
                    block_norms = torch.norm(blocks, dim=(1, 3))
                    
                    # Calculate threshold
                    threshold = torch.quantile(block_norms.flatten(), sparsity)
                    
                    # Create block mask
                    block_mask = (block_norms > threshold).float()
                    
                    # Expand mask to match block size
                    mask = block_mask.unsqueeze(1).unsqueeze(3).expand(h_blocks, block_h, w_blocks, block_w)
                    
                    # Reshape mask to match parameter shape
                    mask = mask.reshape(h, w)
                    
            else:
                # Fall back to element-wise for non-2D tensors
                threshold = torch.quantile(torch.abs(param.data.flatten()), sparsity)
                mask = (torch.abs(param.data) > threshold).float()
                
        else:
            raise ValueError(f"Unsupported pruning granularity: {self.config.granularity}")
            
        return mask
    
    def _apply_structured_pruning(
        self,
        model: AutoModelForCausalLM,
        prunable_params: Dict[str, torch.nn.Parameter]
    ):
        """
        Apply structured pruning to the model.
        
        Args:
            model: Model to prune
            prunable_params: Dictionary of prunable parameters
        """
        self.logger.info("Applying structured pruning")
        
        # Group parameters by layer type
        layer_groups = {}
        
        for name, param in prunable_params.items():
            # Extract layer type from name
            if "query" in name or "key" in name or "value" in name:
                layer_type = "attention"
            elif "dense" in name or "fc" in name:
                layer_type = "ffn"
            else:
                layer_type = "other"
                
            # Add to group
            if layer_type not in layer_groups:
                layer_groups[layer_type] = {}
                
            layer_groups[layer_type][name] = param
            
        # Apply structured pruning to each group
        for layer_type, params in layer_groups.items():
            self.logger.info(f"Pruning {layer_type} layers ({len(params)} parameters)")
            
            # For attention layers, we can prune attention heads
            if layer_type == "attention" and len(params) >= 3:
                self._prune_attention_heads(model, params)
                
            # For FFN layers, we can prune neurons
            elif layer_type == "ffn" and len(params) >= 2:
                self._prune_ffn_neurons(params)
                
            # For other layers, fall back to magnitude pruning
            else:
                for name, param in params.items():
                    mask = self._create_magnitude_mask(param, self.config.sparsity)
                    param.data = param.data * mask
                    param._prune_mask = mask
    
    def _prune_attention_heads(
        self,
        model: AutoModelForCausalLM,
        attention_params: Dict[str, torch.nn.Parameter]
    ):
        """
        Prune attention heads.
        
        Args:
            model: Model to prune
            attention_params: Dictionary of attention parameters
        """
        # This is a simplified implementation and may need adaptation for specific model architectures
        
        # Group parameters by layer
        layer_params = {}
        
        for name, param in attention_params.items():
            # Extract layer index from name
            # This assumes a naming convention like "layers.0.attention.query"
            parts = name.split(".")
            layer_idx = None
            
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_idx = int(part)
                    break
                    
            if layer_idx is None:
                continue
                
            # Add to layer group
            if layer_idx not in layer_params:
                layer_params[layer_idx] = {}
                
   
(Content truncated due to size limit. Use line ranges to read in chunks)