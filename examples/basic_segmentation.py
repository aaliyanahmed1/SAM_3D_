"""
Basic image segmentation example using SAM 3D.

This example demonstrates:
- Loading a SAM model
- Point-based segmentation
- Box-based segmentation
- Visualizing results
"""

from sam3d import SAM3DSegmentor
from sam3d.utils import visualize_segmentation
import numpy as np


def main():
    """Main example function."""
    print("=" * 60)
    print("SAM 3D - Basic Segmentation Example")
    print("=" * 60)
    
    # Initialize segmentor
    print("\n1. Initializing SAM 3D Segmentor...")
    segmentor = SAM3DSegmentor(
        model_type='vit_h',  # Options: vit_h, vit_l, vit_b
        device='cuda'         # Options: cuda, cpu
    )
    
    # Load model
    print("\n2. Loading model...")
    print("Note: Download checkpoint from:")
    print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    
    # Uncomment when you have the checkpoint:
    # segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')
    
    # Example image path (replace with your image)
    image_path = 'path/to/your/image.jpg'
    
    # Example 1: Point-based segmentation
    print("\n3. Example 1: Point-based Segmentation")
    print("Click points on the object you want to segment")
    
    # Define points (x, y coordinates)
    points = [
        [300, 200],  # Foreground point
        [350, 250],  # Another foreground point
    ]
    labels = [1, 1]  # 1 = foreground, 0 = background
    
    print(f"Points: {points}")
    print(f"Labels: {labels}")
    
    # Uncomment when model is loaded:
    # masks, scores, image = segmentor.segment_with_points(
    #     image=image_path,
    #     points=points,
    #     labels=labels
    # )
    # print(f"Generated {len(masks)} masks")
    # print(f"Scores: {scores}")
    
    # Visualize results
    # visualize_segmentation(
    #     image=image,
    #     masks=masks,
    #     scores=scores,
    #     save_path='outputs/segmentation_points.png'
    # )
    
    # Example 2: Box-based segmentation
    print("\n4. Example 2: Box-based Segmentation")
    print("Draw a bounding box around the object")
    
    # Define bounding box [x1, y1, x2, y2]
    box = [100, 100, 500, 500]
    
    print(f"Box: {box}")
    
    # Uncomment when model is loaded:
    # masks, scores, image = segmentor.segment_with_box(
    #     image=image_path,
    #     box=box
    # )
    # print(f"Generated {len(masks)} masks")
    
    # Visualize results
    # visualize_segmentation(
    #     image=image,
    #     masks=masks,
    #     scores=scores,
    #     save_path='outputs/segmentation_box.png'
    # )
    
    # Example 3: Combining points (foreground + background)
    print("\n5. Example 3: Foreground + Background Points")
    
    points_combined = [
        [300, 200],  # Foreground
        [350, 250],  # Foreground
        [50, 50],    # Background
    ]
    labels_combined = [1, 1, 0]
    
    print(f"Points: {points_combined}")
    print(f"Labels: {labels_combined}")
    
    # Uncomment when model is loaded:
    # masks, scores, image = segmentor.segment_with_points(
    #     image=image_path,
    #     points=points_combined,
    #     labels=labels_combined
    # )
    
    # Example 4: Text-based segmentation (SAM 3 only)
    print("\n6. Example 4: Text-based Segmentation (SAM 3)")
    
    text_prompt = "red car"
    print(f"Text prompt: '{text_prompt}'")
    
    # Uncomment when model is loaded:
    # try:
    #     masks, scores, image = segmentor.segment_with_text(
    #         image=image_path,
    #         text_prompt=text_prompt
    #     )
    #     print("Text-based segmentation successful!")
    # except RuntimeError as e:
    #     print(f"Text segmentation not available: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

