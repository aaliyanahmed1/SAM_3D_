"""I/O utilities for SAM 3D."""

import numpy as np
import yaml
from PIL import Image
from pathlib import Path
from typing import Union, Dict, Any
import cv2


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array (H, W, 3)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    image = Image.open(path).convert('RGB')
    return np.array(image)


def save_mask(mask: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save segmentation mask.
    
    Args:
        mask: Binary mask (H, W)
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    mask_img = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(path)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_video_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get video information.
    
    Args:
        path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    
    cap = cv2.VideoCapture(str(path))
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    
    return info

