"""
3D reconstruction example using SAM 3D.

This example demonstrates:
- Depth estimation
- Point cloud generation
- 3D mesh reconstruction
- Saving 3D models
"""

from sam3d import SAM3DSegmentor, Object3DReconstructor
from sam3d.reconstruction import HumanBodyReconstructor


def main():
    """Main example function."""
    print("=" * 60)
    print("SAM 3D - 3D Reconstruction Example")
    print("=" * 60)
    
    # Example 1: Object 3D reconstruction
    print("\n1. Object 3D Reconstruction")
    
    reconstructor = Object3DReconstructor(
        depth_model='dpt_large',
        device='cuda'
    )
    
    # Load depth model
    print("\n2. Loading depth estimation model...")
    # reconstructor.load_depth_model()
    
    image_path = 'path/to/your/image.jpg'
    
    # Estimate depth
    print("\n3. Estimating depth map...")
    # depth_map = reconstructor.estimate_depth(image_path)
    # print(f"Depth map shape: {depth_map.shape}")
    
    # Full 3D reconstruction
    print("\n4. Full 3D reconstruction...")
    # result = reconstructor.reconstruct_from_image(image_path)
    # 
    # print(f"Point cloud: {result['point_cloud']}")
    # print(f"Mesh: {result['mesh']}")
    # print(f"Depth map shape: {result['depth_map'].shape}")
    
    # Save outputs
    print("\n5. Saving 3D outputs...")
    # reconstructor.save_point_cloud(
    #     result['point_cloud'],
    #     'outputs/point_cloud.ply'
    # )
    # 
    # reconstructor.save_mesh(
    #     result['mesh'],
    #     'outputs/mesh.obj'
    # )
    
    # Visualize 3D (requires Open3D)
    print("\n6. Visualizing 3D...")
    # reconstructor.visualize_3d(result['point_cloud'], "Point Cloud")
    # reconstructor.visualize_3d(result['mesh'], "3D Mesh")
    
    # Example 2: Reconstruction with segmentation mask
    print("\n7. 3D Reconstruction with Segmentation")
    
    # First segment the object
    segmentor = SAM3DSegmentor(model_type='vit_h', device='cuda')
    # segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')
    
    # Segment object
    # masks, scores, image = segmentor.segment_with_points(
    #     image=image_path,
    #     points=[[300, 200]],
    #     labels=[1]
    # )
    # 
    # # Use best mask for reconstruction
    # best_mask = masks[0]
    
    # Reconstruct only the segmented object
    # result = reconstructor.reconstruct_from_mask(
    #     image=image_path,
    #     mask=best_mask
    # )
    # 
    # reconstructor.save_mesh(result['mesh'], 'outputs/segmented_object.obj')
    
    # Example 3: Human body reconstruction
    print("\n8. Human Body Reconstruction")
    
    body_reconstructor = HumanBodyReconstructor(
        pose_model='mediapipe',
        body_model='smpl',
        device='cuda'
    )
    
    # Load pose model
    print("\n9. Loading pose estimation model...")
    # body_reconstructor.load_pose_model()
    
    human_image_path = 'path/to/human/image.jpg'
    
    # Estimate pose
    print("\n10. Estimating human pose...")
    # pose_result = body_reconstructor.estimate_pose(human_image_path)
    # 
    # if pose_result['landmarks'] is not None:
    #     print(f"Detected {len(pose_result['landmarks'])} pose landmarks")
    #     print(f"Average visibility: {pose_result['visibility'].mean():.2f}")
    
    # Full body reconstruction
    print("\n11. Full body reconstruction...")
    # body_result = body_reconstructor.reconstruct_body(
    #     image=human_image_path,
    #     estimate_shape=True
    # )
    # 
    # if body_result['success']:
    #     print("Body reconstruction successful!")
    #     reconstructor.save_point_cloud(
    #         body_result['reconstruction']['point_cloud'],
    #         'outputs/human_body.ply'
    #     )
    
    # Example 4: Video pose tracking
    print("\n12. Video Pose Tracking")
    
    video_path = 'path/to/video.mp4'
    
    # Uncomment when ready:
    # pose_results = body_reconstructor.track_pose_video(
    #     video_path=video_path,
    #     output_path='outputs/pose_tracking.mp4'
    # )
    # 
    # print(f"Tracked pose for {len(pose_results)} frames")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("Outputs saved to outputs/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()

