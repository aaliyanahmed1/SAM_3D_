"""Integration tests for full workflows."""

import pytest
import numpy as np
from pathlib import Path

from sam3d import SAM3DSegmentor, Object3DReconstructor


@pytest.mark.integration
class TestFullWorkflow:
    """Test complete workflows from segmentation to 3D."""
    
    @pytest.mark.skip(reason="Requires model checkpoints")
    def test_segment_and_reconstruct(self):
        """Test segmentation followed by 3D reconstruction."""
        # Create dummy image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Segment
        segmentor = SAM3DSegmentor(model_type='vit_h', device='cpu')
        segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')
        
        masks, scores, _ = segmentor.segment_with_points(
            image=image,
            points=[[256, 256]],
            labels=[1]
        )
        
        # Reconstruct
        reconstructor = Object3DReconstructor(device='cpu')
        reconstructor.load_depth_model()
        
        result = reconstructor.reconstruct_from_mask(
            image=image,
            mask=masks[0]
        )
        
        assert result is not None
        assert 'point_cloud' in result
        assert 'mesh' in result
    
    @pytest.mark.skip(reason="Requires model and video")
    def test_video_tracking_workflow(self):
        """Test video tracking workflow."""
        from sam3d.segmentation import VideoSegmentor
        
        segmentor = SAM3DSegmentor(model_type='vit_h', device='cpu')
        segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')
        
        video_segmentor = VideoSegmentor(segmentor=segmentor)
        
        masks = video_segmentor.segment_video(
            video_path='test_video.mp4',
            initial_prompt={'points': [[100, 100]], 'labels': [1]},
            max_frames=10
        )
        
        assert len(masks) > 0
        assert all(isinstance(m, np.ndarray) for m in masks)

