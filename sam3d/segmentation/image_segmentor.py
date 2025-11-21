"""Image segmentation using SAM 3D."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from loguru import logger

from sam3d.core import Config, ModelLoader


class SAM3DSegmentor:
    """
    Image segmentation using Segment Anything Model (SAM).
    
    Supports multiple prompting methods:
    - Point prompts (foreground/background)
    - Bounding box prompts
    - Text prompts (SAM 3 only)
    - Mask prompts (for refinement)
    """
    
    def __init__(
        self,
        model_type: str = "vit_h",
        device: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize segmentor.
        
        Args:
            model_type: Model variant (vit_h, vit_l, vit_b)
            device: Device to run on (cuda, cpu, or None for auto)
            config: Optional configuration object
        """
        self.config = config or Config()
        self.model_type = model_type
        self.device = device or self.config.model.device
        
        self.loader = ModelLoader(model_type=model_type, device=self.device)
        self.model = None
        self.predictor = None
        self.processor = None
        
        logger.info(f"Initialized SAM3DSegmentor with {model_type} on {self.device}")
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load the SAM model.
        
        Args:
            checkpoint_path: Path to checkpoint file (optional)
        """
        logger.info("Loading SAM model...")
        
        try:
            self.model = self.loader.load_auto(checkpoint_path)
            self.predictor = self.loader.get_predictor()
            
            # Try to get processor for text-based segmentation
            try:
                self.processor = self.loader.get_processor()
            except RuntimeError:
                logger.debug("Processor not available (checkpoint-based model)")
            
            # Optimize for inference
            self.loader.optimize_for_inference()
            
            # Log model info
            info = self.loader.get_model_info()
            logger.success(f"Model loaded: {info['total_parameters']:,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Load and convert image to numpy array."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
            return np.array(image)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
    
    def segment_with_points(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        points: List[List[int]],
        labels: List[int],
        multimask_output: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment objects using point prompts.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            points: List of [x, y] coordinates
            labels: List of labels (1=foreground, 0=background)
            multimask_output: Whether to output multiple masks
            
        Returns:
            Tuple of (masks, scores, image_array)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image
        image_np = self._load_image(image)
        
        # Set image for predictor
        self.predictor.set_image(image_np)
        
        # Configure output
        if multimask_output is None:
            multimask_output = self.config.segmentation.multimask_output
        
        # Predict
        points_array = np.array(points)
        labels_array = np.array(labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=multimask_output,
        )
        
        logger.debug(f"Generated {len(masks)} masks with scores: {scores}")
        
        return masks, scores, image_np
    
    def segment_with_box(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        box: List[int],
        multimask_output: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment object using bounding box.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            box: Bounding box [x1, y1, x2, y2]
            multimask_output: Whether to output multiple masks
            
        Returns:
            Tuple of (masks, scores, image_array)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image
        image_np = self._load_image(image)
        
        # Set image for predictor
        self.predictor.set_image(image_np)
        
        # Configure output
        if multimask_output is None:
            multimask_output = self.config.segmentation.multimask_output
        
        # Predict
        box_array = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_array,
            multimask_output=multimask_output,
        )
        
        logger.debug(f"Generated {len(masks)} masks with scores: {scores}")
        
        return masks, scores, image_np
    
    def segment_with_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Segment objects using text prompts (SAM 3 feature).
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            text_prompt: Text description of object to segment
            
        Returns:
            Tuple of (masks, scores, image_array)
        """
        if self.processor is None:
            raise RuntimeError(
                "Text-based segmentation requires HuggingFace model. "
                "Load with: loader.load_from_huggingface()"
            )
        
        # Load image
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        image_np = np.array(image_pil)
        
        # Process with text prompt
        inputs = self.processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            masks = outputs.pred_masks.cpu().numpy()
            
            # Try to get scores if available
            try:
                scores = outputs.iou_scores.cpu().numpy()
            except AttributeError:
                scores = None
        
        logger.debug(f"Text prompt '{text_prompt}' generated {len(masks)} masks")
        
        return masks, scores, image_np
    
    def segment_everything(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Automatically segment everything in the image.
        
        Args:
            image: Input image
            points_per_side: Number of points per side for grid
            pred_iou_thresh: IoU threshold for predictions
            stability_score_thresh: Stability score threshold
            
        Returns:
            List of segmentation dictionaries
        """
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("segment-anything package required for automatic segmentation")
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image
        image_np = self._load_image(image)
        
        # Create automatic mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        
        # Generate masks
        logger.info("Generating automatic segmentation...")
        masks = mask_generator.generate(image_np)
        
        logger.success(f"Generated {len(masks)} object masks")
        
        return masks
    
    def refine_mask(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: np.ndarray,
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine an existing mask with additional prompts.
        
        Args:
            image: Input image
            mask: Existing mask to refine
            points: Additional point prompts
            labels: Labels for points
            
        Returns:
            Tuple of (refined_masks, scores)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image
        image_np = self._load_image(image)
        
        # Set image for predictor
        self.predictor.set_image(image_np)
        
        # Prepare inputs
        point_coords = np.array(points) if points else None
        point_labels = np.array(labels) if labels else None
        
        # Predict with mask input
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask[None, :, :],
            multimask_output=True,
        )
        
        logger.debug(f"Refined mask with scores: {scores}")
        
        return masks, scores
    
    def batch_segment(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompts: List[Dict[str, Any]],
        batch_size: int = 4
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Batch segmentation for multiple images.
        
        Args:
            images: List of input images
            prompts: List of prompt dictionaries
            batch_size: Processing batch size
            
        Returns:
            List of (masks, scores) tuples
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            
            for img, prompt in zip(batch_images, batch_prompts):
                if 'points' in prompt:
                    masks, scores, _ = self.segment_with_points(
                        img, prompt['points'], prompt['labels']
                    )
                elif 'box' in prompt:
                    masks, scores, _ = self.segment_with_box(img, prompt['box'])
                elif 'text' in prompt:
                    masks, scores, _ = self.segment_with_text(img, prompt['text'])
                else:
                    raise ValueError(f"Invalid prompt: {prompt}")
                
                results.append((masks, scores))
        
        logger.info(f"Batch processed {len(images)} images")
        
        return results

