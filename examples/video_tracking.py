"""
Video object tracking example using SAM 3D.

This example demonstrates:
- Tracking a single object in video
- Tracking multiple objects
- Extracting frames
- Creating visualization videos
"""

from sam3d import SAM3DSegmentor, VideoObjectTracker
from sam3d.segmentation import VideoSegmentor


def main():
    """Main example function."""
    print("=" * 60)
    print("SAM 3D - Video Object Tracking Example")
    print("=" * 60)
    
    # Initialize segmentor
    print("\n1. Initializing SAM 3D...")
    segmentor = SAM3DSegmentor(model_type='vit_h', device='cuda')
    
    # Load model
    print("\n2. Loading model...")
    # segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')
    
    # Initialize video segmentor
    video_segmentor = VideoSegmentor(
        segmentor=segmentor,
        propagation_window=8,
        confidence_threshold=0.5
    )
    
    # Example video path
    video_path = 'path/to/your/video.mp4'
    
    # Example 1: Track single object with points
    print("\n3. Example 1: Track Single Object")
    
    initial_prompt = {
        'points': [[300, 200]],
        'labels': [1]
    }
    
    print(f"Initial prompt: {initial_prompt}")
    
    # Uncomment when ready:
    # masks = video_segmentor.segment_video(
    #     video_path=video_path,
    #     initial_prompt=initial_prompt,
    #     output_path='outputs/tracked_video.mp4',
    #     max_frames=None,  # Process all frames
    #     show_progress=True
    # )
    # print(f"Tracked object for {len(masks)} frames")
    
    # Example 2: Track with bounding box
    print("\n4. Example 2: Track with Bounding Box")
    
    initial_box = {
        'box': [100, 100, 400, 400]
    }
    
    print(f"Initial box: {initial_box}")
    
    # Uncomment when ready:
    # masks = video_segmentor.segment_video(
    #     video_path=video_path,
    #     initial_prompt=initial_box,
    #     output_path='outputs/tracked_box_video.mp4'
    # )
    
    # Example 3: Track multiple objects
    print("\n5. Example 3: Track Multiple Objects")
    
    object_prompts = [
        {'points': [[100, 100]], 'labels': [1]},  # Object 1
        {'points': [[300, 300]], 'labels': [1]},  # Object 2
        {'box': [500, 200, 700, 400]},            # Object 3
    ]
    
    print(f"Tracking {len(object_prompts)} objects")
    
    # Uncomment when ready:
    # all_masks = video_segmentor.segment_multi_object(
    #     video_path=video_path,
    #     object_prompts=object_prompts,
    #     output_path='outputs/multi_object_tracking.mp4'
    # )
    # print(f"Tracked {len(all_masks)} objects")
    
    # Example 4: Extract frames
    print("\n6. Example 4: Extract Frames from Video")
    
    # Uncomment when ready:
    # frame_paths = video_segmentor.extract_frames(
    #     video_path=video_path,
    #     output_dir='outputs/frames/',
    #     frame_rate=10  # Extract every 10th frame
    # )
    # print(f"Extracted {len(frame_paths)} frames")
    
    # Example 5: Advanced tracking with VideoObjectTracker
    print("\n7. Example 5: Advanced Multi-Object Tracking")
    
    tracker = VideoObjectTracker(
        segmentor=segmentor,
        max_objects=10,
        min_confidence=0.5
    )
    
    # Uncomment when ready:
    # track_result = tracker.track_object(
    #     video_path=video_path,
    #     initial_prompt={'points': [[200, 200]], 'labels': [1]},
    #     output_path='outputs/advanced_tracking.mp4'
    # )
    # 
    # print(f"Track ID: {track_result['track_id']}")
    # print(f"Frames tracked: {track_result['num_frames']}")
    # print(f"Bounding boxes: {len(track_result['bounding_boxes'])}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

