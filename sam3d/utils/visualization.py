"""Visualization utilities for SAM 3D."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import List, Optional, Union, Tuple
import cv2


def visualize_segmentation(
    image: np.ndarray,
    masks: np.ndarray,
    scores: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Visualize segmentation results.
    
    Args:
        image: Original image (H, W, 3)
        masks: Segmentation masks (N, H, W)
        scores: Optional confidence scores (N,)
        save_path: Optional path to save visualization
        show: Whether to display visualization
    """
    num_masks = min(len(masks), 4)
    
    fig, axes = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))
    
    if num_masks == 0:
        axes = [axes]
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show masks
    for idx in range(num_masks):
        axes[idx + 1].imshow(image)
        axes[idx + 1].imshow(masks[idx], alpha=0.5, cmap='jet')
        
        title = f'Mask {idx + 1}'
        if scores is not None:
            title += f' (Score: {scores[idx]:.3f})'
        axes[idx + 1].set_title(title)
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_video_from_masks(
    video_path: Union[str, Path],
    masks: List[np.ndarray],
    output_path: Union[str, Path],
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> None:
    """
    Create visualization video with mask overlay.
    
    Args:
        video_path: Path to original video
        masks: List of masks for each frame
        output_path: Path to save output video
        color: Overlay color (B, G, R)
        alpha: Transparency (0-1)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(masks):
            break
        
        # Overlay mask
        mask = masks[frame_idx]
        overlay = frame.copy()
        overlay[mask > 0] = color
        output = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        writer.write(output)
        frame_idx += 1
    
    cap.release()
    writer.release()


def show_anns(anns: List[dict], ax: Optional[plt.Axes] = None):
    """
    Show automatic segmentation annotations.
    
    Args:
        anns: List of annotation dictionaries from automatic segmentation
        ax: Matplotlib axes (creates new if None)
    """
    if ax is None:
        ax = plt.gca()
    
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    ax.imshow(img)

