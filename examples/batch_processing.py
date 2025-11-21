"""
Batch processing example for production use.

This example demonstrates:
- Batch image segmentation
- Parallel processing
- Progress tracking
- Error handling
- Results aggregation
"""

from sam3d import SAM3DSegmentor
from pathlib import Path
from typing import List
import json
from tqdm import tqdm


def batch_segment_images(
    image_paths: List[str],
    output_dir: str,
    checkpoint_path: str,
    batch_size: int = 4
):
    """
    Batch segment images with error handling.
    
    Args:
        image_paths: List of image paths
        output_dir: Directory to save results
        checkpoint_path: Path to SAM checkpoint
        batch_size: Processing batch size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize segmentor
    print("Initializing SAM 3D...")
    segmentor = SAM3DSegmentor(model_type='vit_h', device='cuda')
    segmentor.load_model(checkpoint_path)
    
    # Prepare prompts for each image
    # In production, these would come from user input or database
    prompts = []
    for _ in image_paths:
        # Example: center point prompt
        prompts.append({
            'points': [[512, 512]],  # Adjust based on your images
            'labels': [1]
        })
    
    # Process in batches
    results = []
    errors = []
    
    print(f"\nProcessing {len(image_paths)} images in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_images = image_paths[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        
        for img_path, prompt in zip(batch_images, batch_prompts):
            try:
                # Segment image
                masks, scores, image = segmentor.segment_with_points(
                    image=img_path,
                    points=prompt['points'],
                    labels=prompt['labels']
                )
                
                # Save best mask
                best_idx = scores.argmax()
                mask = masks[best_idx]
                score = scores[best_idx]
                
                # Save mask
                img_name = Path(img_path).stem
                mask_path = output_dir / f"{img_name}_mask.npy"
                import numpy as np
                np.save(mask_path, mask)
                
                # Record result
                results.append({
                    'image': img_path,
                    'mask_path': str(mask_path),
                    'score': float(score),
                    'success': True
                })
                
            except Exception as e:
                # Record error
                errors.append({
                    'image': img_path,
                    'error': str(e),
                    'success': False
                })
    
    # Save results summary
    summary = {
        'total': len(image_paths),
        'successful': len(results),
        'failed': len(errors),
        'results': results,
        'errors': errors
    }
    
    summary_path = output_dir / 'results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Processed {len(results)}/{len(image_paths)} images successfully")
    print(f"❌ Failed: {len(errors)}")
    print(f"📊 Summary saved to: {summary_path}")
    
    return summary


def main():
    """Main example function."""
    print("=" * 60)
    print("SAM 3D - Batch Processing Example")
    print("=" * 60)
    
    # Example configuration
    image_directory = 'path/to/images/'
    output_directory = 'outputs/batch_results/'
    checkpoint_path = 'checkpoints/sam_vit_h_4b8939.pth'
    
    # Get all images
    image_dir = Path(image_directory)
    image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    print(f"\nFound {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found! Update image_directory path.")
        return
    
    # Process batch
    # Uncomment when ready:
    # summary = batch_segment_images(
    #     image_paths=[str(p) for p in image_paths],
    #     output_dir=output_directory,
    #     checkpoint_path=checkpoint_path,
    #     batch_size=8
    # )
    
    # Print statistics
    # print("\n" + "=" * 60)
    # print("Processing Statistics")
    # print("=" * 60)
    # print(f"Total images: {summary['total']}")
    # print(f"Successful: {summary['successful']}")
    # print(f"Failed: {summary['failed']}")
    # print(f"Success rate: {summary['successful']/summary['total']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

