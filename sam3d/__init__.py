"""
SAM 3D - Segment Anything Model 3D
Production-grade implementation for image segmentation, 3D reconstruction, and video tracking.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from sam3d.core.config import Config
from sam3d.segmentation.image_segmentor import SAM3DSegmentor
from sam3d.reconstruction.object_3d import Object3DReconstructor
from sam3d.tracking.tracker import VideoObjectTracker

__all__ = [
    "__version__",
    "Config",
    "SAM3DSegmentor",
    "Object3DReconstructor",
    "VideoObjectTracker",
]

