"""Video segmentation using SAM 3D."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from loguru import logger
from tqdm import tqdm

from sam3d.segmentation.image_segmentor import SAM3DSegmentor


class VideoSegmentor:
    """
    Video segmentation with temporal consistency.
    
    Features:
    - Frame-by-frame segmentation
    - Temporal mask propagation
    - Track-based segmentation
    - Multi-object tracking
    """
    
    def __init__(
        self,
        segmentor: Optional[SAM3DSegmentor] = None,
        propagation_window: int = 8,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize video segmentor.
        
        Args:
            segmentor: SAM3D segmentor instance (creates new if None)
            propagation_window: Number of frames for temporal propagation
            confidence_threshold: Minimum confidence for mask propagation
        """
        self.segmentor = segmentor or SAM3DSegmentor()
        self.propagation_window = propagation_window
        self.confidence_threshold = confidence_threshold
        
        logger.info("Initialized VideoSegmentor")
    
    def segment_video(
        self,
        video_path: Union[str, Path],
        initial_prompt: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Segment object throughout video.
        
        Args:
            video_path: Path to input video
            initial_prompt: Initial segmentation prompt (points, box, or text)
            output_path: Optional path to save output video
            max_frames: Maximum frames to process
            show_progress: Show progress bar
            
        Returns:
            List of masks for each frame
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output writer if requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        masks = []
        prev_mask = None
        
        pbar = tqdm(total=total_frames, disable=not show_progress, desc="Processing frames")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # First frame: use initial prompt
            if frame_idx == 0:
                mask, score = self._segment_frame(frame_rgb, initial_prompt)
                prev_mask = mask
            else:
                # Subsequent frames: use mask propagation
                mask, score = self._propagate_mask(frame_rgb, prev_mask)
                
                # If confidence too low, re-segment with prompt
                if score < self.confidence_threshold:
                    logger.warning(f"Low confidence at frame {frame_idx}, re-segmenting")
                    mask, score = self._segment_frame(frame_rgb, initial_prompt)
                
                prev_mask = mask
            
            masks.append(mask)
            
            # Write output frame if requested
            if writer:
                output_frame = self._overlay_mask(frame, mask)
                writer.write(output_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if writer:
            writer.release()
            logger.success(f"Saved output video: {output_path}")
        
        logger.success(f"Processed {len(masks)} frames")
        
        return masks
    
    def _segment_frame(
        self,
        frame: np.ndarray,
        prompt: Dict[str, Any]
    ) -> Tuple[np.ndarray, float]:
        """Segment a single frame with given prompt."""
        if 'points' in prompt:
            masks, scores, _ = self.segmentor.segment_with_points(
                frame, prompt['points'], prompt['labels']
            )
        elif 'box' in prompt:
            masks, scores, _ = self.segmentor.segment_with_box(frame, prompt['box'])
        elif 'text' in prompt:
            masks, scores, _ = self.segmentor.segment_with_text(frame, prompt['text'])
        else:
            raise ValueError(f"Invalid prompt: {prompt}")
        
        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def _propagate_mask(
        self,
        frame: np.ndarray,
        prev_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Propagate mask from previous frame."""
        # Use previous mask as prompt for refinement
        masks, scores = self.segmentor.refine_mask(frame, prev_mask)
        
        # Return best refined mask
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def _overlay_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay mask on frame."""
        overlay = frame.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        frame_rate: Optional[int] = None
    ) -> List[Path]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video
            output_dir: Directory to save frames
            frame_rate: Extract every Nth frame (None for all)
            
        Returns:
            List of saved frame paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frame_paths = []
        frame_idx = 0
        saved_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should save this frame
            if frame_rate is None or frame_idx % frame_rate == 0:
                frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
                saved_idx += 1
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
        
        return frame_paths
    
    def create_video_from_frames(
        self,
        frame_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        fps: int = 30
    ):
        """
        Create video from frame sequence.
        
        Args:
            frame_paths: List of frame image paths
            output_path: Output video path
            fps: Frames per second
        """
        if not frame_paths:
            raise ValueError("No frames provided")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write frames
        for frame_path in tqdm(frame_paths, desc="Creating video"):
            frame = cv2.imread(str(frame_path))
            writer.write(frame)
        
        writer.release()
        logger.success(f"Created video: {output_path}")
    
    def segment_multi_object(
        self,
        video_path: Union[str, Path],
        object_prompts: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> List[List[np.ndarray]]:
        """
        Segment multiple objects in video.
        
        Args:
            video_path: Path to video
            object_prompts: List of prompts for each object
            output_path: Optional output video path
            
        Returns:
            List of mask lists (one per object)
        """
        logger.info(f"Tracking {len(object_prompts)} objects")
        
        # Segment each object separately
        all_masks = []
        for obj_idx, prompt in enumerate(object_prompts):
            logger.info(f"Processing object {obj_idx + 1}/{len(object_prompts)}")
            masks = self.segment_video(
                video_path, prompt, output_path=None, show_progress=True
            )
            all_masks.append(masks)
        
        # If output requested, create combined visualization
        if output_path:
            self._create_multi_object_video(video_path, all_masks, output_path)
        
        return all_masks
    
    def _create_multi_object_video(
        self,
        video_path: Union[str, Path],
        all_masks: List[List[np.ndarray]],
        output_path: Union[str, Path]
    ):
        """Create video with multiple object masks overlaid."""
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Define colors for each object
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output_frame = frame.copy()
            
            # Overlay each object's mask
            for obj_idx, masks in enumerate(all_masks):
                if frame_idx < len(masks):
                    color = colors[obj_idx % len(colors)]
                    output_frame = self._overlay_mask(
                        output_frame, masks[frame_idx], color, alpha=0.3
                    )
            
            writer.write(output_frame)
            frame_idx += 1
        
        cap.release()
        writer.release()
        logger.success(f"Created multi-object video: {output_path}")

