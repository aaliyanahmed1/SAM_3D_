"""Model loading utilities for SAM 3D."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import warnings

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("segment-anything package not installed")

try:
    from transformers import SamModel, SamProcessor
    TRANSFORMERS_SAM_AVAILABLE = True
except ImportError:
    TRANSFORMERS_SAM_AVAILABLE = False
    logger.warning("transformers SAM support not available")


class ModelLoader:
    """Handles loading of SAM models from various sources."""
    
    # Model checkpoint URLs
    CHECKPOINT_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    
    # HuggingFace model names
    HF_MODEL_NAMES = {
        "vit_h": "facebook/sam-vit-huge",
        "vit_l": "facebook/sam-vit-large",
        "vit_b": "facebook/sam-vit-base",
    }
    
    def __init__(self, model_type: str = "vit_h", device: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            model_type: Type of SAM model (vit_h, vit_l, vit_b)
            device: Device to load model on (cuda, cpu, or None for auto)
        """
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.predictor = None
        self.processor = None
        
        logger.info(f"Initialized ModelLoader with type={model_type}, device={self.device}")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> Any:
        """
        Load SAM model from local checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything package required. Install with: pip install segment-anything")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        try:
            sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
            sam.to(device=self.device)
            sam.eval()
            
            self.model = sam
            self.predictor = SamPredictor(sam)
            
            logger.success(f"Successfully loaded model from {checkpoint_path}")
            return sam
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_from_huggingface(self, model_name: Optional[str] = None) -> Any:
        """
        Load SAM model from HuggingFace.
        
        Args:
            model_name: HuggingFace model name (optional, uses default for model_type)
            
        Returns:
            Loaded model
        """
        if not TRANSFORMERS_SAM_AVAILABLE:
            raise ImportError("transformers package required. Install with: pip install transformers")
        
        model_name = model_name or self.HF_MODEL_NAMES.get(self.model_type)
        if not model_name:
            raise ValueError(f"No HuggingFace model available for type: {self.model_type}")
        
        logger.info(f"Loading model from HuggingFace: {model_name}")
        
        try:
            self.model = SamModel.from_pretrained(model_name)
            self.processor = SamProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.success(f"Successfully loaded model from HuggingFace: {model_name}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model from HuggingFace: {e}")
            raise
    
    def load_auto(self, checkpoint_path: Optional[str] = None) -> Any:
        """
        Automatically load model from best available source.
        
        Args:
            checkpoint_path: Optional local checkpoint path
            
        Returns:
            Loaded model
        """
        # Try local checkpoint first
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info("Loading from local checkpoint")
            return self.load_from_checkpoint(checkpoint_path)
        
        # Try HuggingFace
        if TRANSFORMERS_SAM_AVAILABLE:
            try:
                logger.info("Attempting to load from HuggingFace")
                return self.load_from_huggingface()
            except Exception as e:
                logger.warning(f"HuggingFace loading failed: {e}")
        
        # Provide instructions if nothing works
        raise RuntimeError(
            f"Could not load model. Please:\n"
            f"1. Download checkpoint from: {self.CHECKPOINT_URLS[self.model_type]}\n"
            f"2. Or install transformers: pip install transformers\n"
            f"3. Then load with checkpoint_path parameter"
        )
    
    def get_predictor(self) -> Any:
        """Get predictor instance."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_* method first.")
        return self.predictor
    
    def get_processor(self) -> Any:
        """Get processor instance (for HuggingFace models)."""
        if self.processor is None:
            raise RuntimeError("Processor not available. Use HuggingFace model loading.")
        return self.processor
    
    def optimize_for_inference(self):
        """Optimize model for inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        logger.info("Optimizing model for inference")
        
        # Set to eval mode
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Try to use torch.compile (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                logger.info("Applying torch.compile optimization")
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Try half precision if on CUDA
        if self.device == "cuda":
            try:
                logger.info("Converting to half precision (FP16)")
                self.model = self.model.half()
            except Exception as e:
                logger.warning(f"Half precision conversion failed: {e}")
        
        logger.success("Model optimization complete")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "model_type": self.model_type,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_footprint_mb": total_params * 4 / (1024 ** 2),  # Approximate
        }

