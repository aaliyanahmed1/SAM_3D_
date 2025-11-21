"""Unit tests for 3D reconstruction module."""

import pytest
import numpy as np

from sam3d.reconstruction import Object3DReconstructor


@pytest.fixture
def dummy_image():
    """Create a dummy RGB image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_depth():
    """Create a dummy depth map."""
    return np.random.rand(512, 512).astype(np.float32)


@pytest.fixture
def reconstructor():
    """Create reconstructor instance."""
    return Object3DReconstructor(depth_model='dpt_large', device='cpu')


@pytest.mark.unit
class TestObject3DReconstructor:
    """Test Object3DReconstructor class."""
    
    def test_initialization(self):
        """Test reconstructor initialization."""
        reconstructor = Object3DReconstructor(depth_model='dpt_large', device='cpu')
        assert reconstructor.depth_model_name == 'dpt_large'
        assert reconstructor.device == 'cpu'
        assert reconstructor.depth_model is None
    
    def test_depth_to_point_cloud(self, reconstructor, dummy_image, dummy_depth):
        """Test depth to point cloud conversion."""
        point_cloud = reconstructor.depth_to_point_cloud(
            depth_map=dummy_depth,
            image=dummy_image,
            mask=None
        )
        
        assert point_cloud is not None
    
    def test_depth_to_point_cloud_with_mask(self, reconstructor, dummy_image, dummy_depth):
        """Test depth to point cloud with mask."""
        mask = np.random.randint(0, 2, (512, 512), dtype=bool)
        
        point_cloud = reconstructor.depth_to_point_cloud(
            depth_map=dummy_depth,
            image=dummy_image,
            mask=mask
        )
        
        assert point_cloud is not None


@pytest.mark.integration
class TestReconstructionIntegration:
    """Integration tests for reconstruction."""
    
    @pytest.mark.skip(reason="Requires model download")
    def test_full_reconstruction_pipeline(self):
        """Test complete reconstruction pipeline."""
        reconstructor = Object3DReconstructor(depth_model='dpt_large', device='cpu')
        reconstructor.load_depth_model()
        
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        result = reconstructor.reconstruct_from_image(image)
        
        assert 'point_cloud' in result
        assert 'depth_map' in result
        assert result['depth_map'].shape == (512, 512)

