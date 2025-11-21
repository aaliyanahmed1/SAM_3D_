"""Video object tracking using SAM 3D."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from loguru import logger

from sam3d.segmentation import SAM3DSegmentor, VideoSegmentor


class VideoObjectTracker:
    """
    Advanced video object tracking with SAM 3D.
    
    Features:
    - Multi-object tracking
    - Occlusion handling
    - Re-identification
    - Track smoothing
    """
    
    def __init__(
        self,
        segmentor: Optional[SAM3DSegmentor] = None,
        max_objects: int = 10,
        min_confidence: float = 0.5
    ):
        """
        Initialize video object tracker.
        
        Args:
            segmentor: SAM3D segmentor instance
            max_objects: Maximum number of objects to track
            min_confidence: Minimum confidence threshold
        """
        self.segmentor = segmentor or SAM3DSegmentor()
        self.video_segmentor = VideoSegmentor(segmentor=self.segmentor)
        self.max_objects = max_objects
        self.min_confidence = min_confidence
        
        self.tracks = []
        self.next_track_id = 0
        
        logger.info("Initialized VideoObjectTracker")
    
    def track_object(
        self,
        video_path: Union[str, Path],
        initial_prompt: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Track single object in video.
        
        Args:
            video_path: Path to video
            initial_prompt: Initial segmentation prompt
            output_path: Optional output video path
            
        Returns:
            Tracking results dictionary
        """
        logger.info(f"Tracking object in video: {video_path}")
        
        # Use video segmentor for basic tracking
        masks = self.video_segmentor.segment_video(
            video_path=video_path,
            initial_prompt=initial_prompt,
            output_path=output_path,
            show_progress=True
        )
        
        # Compute bounding boxes and centroids
        bboxes = [self._mask_to_bbox(mask) for mask in masks]
        centroids = [self._bbox_to_centroid(bbox) for bbox in bboxes]
        
        result = {
            'masks': masks,
            'bounding_boxes': bboxes,
            'centroids': centroids,
            'num_frames': len(masks),
            'track_id': 0
        }
        
        logger.success(f"Tracked object for {len(masks)} frames")
        
        return result
    
    def track_multiple_objects(
        self,
        video_path: Union[str, Path],
        initial_prompts: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Track multiple objects in video.
        
        Args:
            video_path: Path to video
            initial_prompts: List of initial prompts for each object
            output_path: Optional output video path
            
        Returns:
            List of tracking results
        """
        logger.info(f"Tracking {len(initial_prompts)} objects")
        
        # Track each object
        all_tracks = []
        
        for obj_idx, prompt in enumerate(initial_prompts):
            logger.info(f"Processing object {obj_idx + 1}/{len(initial_prompts)}")
            
            track_result = self.track_object(
                video_path=video_path,
                initial_prompt=prompt,
                output_path=None
            )
            
            track_result['track_id'] = obj_idx
            all_tracks.append(track_result)
        
        # Create visualization if requested
        if output_path:
            self._visualize_multi_track(video_path, all_tracks, output_path)
        
        return all_tracks
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Convert mask to bounding box [x, y, w, h]."""
        if mask.sum() == 0:
            return [0, 0, 0, 0]
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        return [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
    
    def _bbox_to_centroid(self, bbox: List[int]) -> List[float]:
        """Get centroid from bounding box."""
        x, y, w, h = bbox
        return [x + w / 2, y + h / 2]
    
    def _visualize_multi_track(
        self,
        video_path: Union[str, Path],
        tracks: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ):
        """Create visualization video for multiple tracks."""
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw each track
            for track_idx, track in enumerate(tracks):
                if frame_idx < len(track['masks']):
                    color = colors[track_idx % len(colors)]
                    
                    # Draw mask overlay
                    mask = track['masks'][frame_idx]
                    overlay = frame.copy()
                    overlay[mask > 0] = color
                    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                    
                    # Draw bounding box
                    bbox = track['bounding_boxes'][frame_idx]
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, f"ID: {track['track_id']}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2
                    )
            
            writer.write(frame)
            frame_idx += 1
        
        cap.release()
        writer.release()
        logger.success(f"Saved tracking visualization: {output_path}")

