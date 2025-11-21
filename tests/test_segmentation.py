"""Unit tests for segmentation module."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from sam3d.segmentation import SAM3DSegmentor


@pytest.fixture
def dummy_image():
    """Create a dummy RGB image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def segmentor():
    """Create segmentor instance (without loading model)."""
    return SAM3DSegmentor(model_type='vit_h', device='cpu')


@pytest.mark.unit
class TestSAM3DSegmentor:
    """Test SAM3DSegmentor class."""
    
    def test_initialization(self):
        """Test segmentor initialization."""
        segmentor = SAM3DSegmentor(model_type='vit_h', device='cpu')
        assert segmentor.model_type == 'vit_h'
        assert segmentor.device == 'cpu'
        assert segmentor.predictor is None
    
    def test_load_image_from_array(self, segmentor, dummy_image):
        """Test loading image from numpy array."""
        result = segmentor._load_image(dummy_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == dummy_image.shape
    
    def test_load_image_from_pil(self, segmentor, dummy_image):
        """Test loading image from PIL Image."""
        pil_image = Image.fromarray(dummy_image)
        result = segmentor._load_image(pil_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == dummy_image.shape
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        segmentor = SAM3DSegmentor(model_type='invalid')
        # Should raise error when trying to load
        with pytest.raises(Exception):
            segmentor.load_model('fake_checkpoint.pth')
    
    def test_segment_without_model(self, segmentor, dummy_image):
        """Test segmentation without loaded model raises error."""
        with pytest.raises(RuntimeError):
            segmentor.segment_with_points(
                dummy_image,
                points=[[100, 100]],
                labels=[1]
            )


@pytest.mark.integration
class TestSegmentationIntegration:
    """Integration tests for segmentation (requires model)."""
    
    @pytest.mark.skip(reason="Requires model checkpoint")
    def test_full_segmentation_pipeline(self):
        """Test complete segmentation pipeline."""
        segmentor = SAM3DSegmentor(model_type='vit_h', device='cpu')
        segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')
        
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        masks, scores, _ = segmentor.segment_with_points(
            image=image,
            points=[[256, 256]],
            labels=[1]
        )
        
        assert masks is not None
        assert scores is not None
        assert len(masks) > 0
        assert len(scores) == len(masks)

