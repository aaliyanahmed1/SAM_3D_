# SAM 3D Examples

This directory contains comprehensive examples demonstrating all features of SAM 3D.

## 📋 Available Examples

### 1. Basic Segmentation (`basic_segmentation.py`)

Learn the fundamentals of image segmentation with SAM 3D.

**Features:**
- Point-based segmentation
- Box-based segmentation
- Text-based segmentation (SAM 3)
- Combining foreground and background points

**Usage:**
```bash
python examples/basic_segmentation.py
```

**Prerequisites:**
- SAM checkpoint downloaded
- Sample image

### 2. Video Tracking (`video_tracking.py`)

Track objects across video frames with temporal consistency.

**Features:**
- Single object tracking
- Multiple object tracking
- Frame extraction
- Visualization generation

**Usage:**
```bash
python examples/video_tracking.py
```

**Prerequisites:**
- SAM checkpoint downloaded
- Sample video file

### 3. 3D Reconstruction (`3d_reconstruction.py`)

Reconstruct 3D models from 2D images.

**Features:**
- Depth estimation
- Point cloud generation
- Mesh reconstruction
- Human pose estimation
- Body reconstruction

**Usage:**
```bash
python examples/3d_reconstruction.py
```

**Prerequisites:**
- Depth estimation model (auto-downloaded)
- Open3D for visualization

### 4. Batch Processing (`batch_processing.py`)

Production-ready batch processing with error handling.

**Features:**
- Batch image processing
- Progress tracking
- Error handling
- Results aggregation
- JSON output

**Usage:**
```bash
python examples/batch_processing.py
```

**Prerequisites:**
- Directory of images to process

## 🚀 Quick Start

### Step 1: Download Model Checkpoints

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download SAM ViT-H (best quality)
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

### Step 2: Prepare Your Data

```bash
# Create data directories
mkdir -p data/images
mkdir -p data/videos
mkdir -p outputs
```

### Step 3: Run Examples

```bash
# Basic segmentation
python examples/basic_segmentation.py

# Video tracking
python examples/video_tracking.py

# 3D reconstruction
python examples/3d_reconstruction.py

# Batch processing
python examples/batch_processing.py
```

## 📝 Example Modifications

### Custom Image Segmentation

```python
from sam3d import SAM3DSegmentor

segmentor = SAM3DSegmentor(model_type='vit_h')
segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')

# Your custom image
masks, scores, image = segmentor.segment_with_points(
    image='my_image.jpg',
    points=[[x, y]],  # Your coordinates
    labels=[1]
)
```

### Custom Video Tracking

```python
from sam3d.segmentation import VideoSegmentor

video_segmentor = VideoSegmentor()
masks = video_segmentor.segment_video(
    video_path='my_video.mp4',
    initial_prompt={'points': [[x, y]], 'labels': [1]},
    output_path='tracked.mp4'
)
```

### Custom 3D Reconstruction

```python
from sam3d.reconstruction import Object3DReconstructor

reconstructor = Object3DReconstructor()
result = reconstructor.reconstruct_from_image('my_image.jpg')
reconstructor.save_mesh(result['mesh'], 'my_model.obj')
```

## 🎯 Use Case Examples

### E-commerce: Product Segmentation

```python
# Segment product from background
masks, scores, _ = segmentor.segment_with_points(
    image='product.jpg',
    points=[[center_x, center_y]],
    labels=[1]
)

# Save with transparent background
# ... use mask to create PNG with alpha channel
```

### Medical Imaging: Organ Segmentation

```python
# Segment specific organ
masks, scores, _ = segmentor.segment_with_box(
    image='scan.jpg',
    box=[x1, y1, x2, y2]  # Region of interest
)
```

### Autonomous Driving: Object Detection

```python
# Track vehicles in dashcam footage
masks = video_segmentor.segment_video(
    video_path='dashcam.mp4',
    initial_prompt={'text': 'car in front'},
    output_path='tracked_vehicle.mp4'
)
```

### AR/VR: 3D Asset Creation

```python
# Create 3D model from photo
result = reconstructor.reconstruct_from_image('object.jpg')
reconstructor.save_mesh(result['mesh'], 'asset.obj')
```

## 🔧 Troubleshooting

### Issue: Model not loading

```python
# Solution: Check checkpoint path
import os
checkpoint_path = 'checkpoints/sam_vit_h_4b8939.pth'
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found: {checkpoint_path}")
    print("Download from: https://github.com/facebookresearch/segment-anything")
```

### Issue: CUDA out of memory

```python
# Solution 1: Use smaller model
segmentor = SAM3DSegmentor(model_type='vit_b')  # Instead of vit_h

# Solution 2: Use CPU
segmentor = SAM3DSegmentor(device='cpu')

# Solution 3: Reduce batch size
batch_size = 2  # Instead of 8
```

### Issue: No object detected

```python
# Solution: Try multiple prompts
points_list = [
    [[100, 100]],
    [[200, 200]],
    [[300, 300]]
]

for points in points_list:
    masks, scores, _ = segmentor.segment_with_points(
        image='image.jpg',
        points=points,
        labels=[1]
    )
    if scores.max() > 0.9:  # Good detection
        break
```

## 📚 Additional Resources

- [API Documentation](../docs/api_reference.md)
- [User Guide](../docs/use_cases.md)
- [GitHub Repository](https://github.com/yourusername/SAM_3D)
- [Video Tutorials](https://youtube.com/...)

## 🤝 Contributing

Found a bug or want to add an example? Please contribute!

1. Fork the repository
2. Create your example
3. Submit a pull request

## 📄 License

These examples are part of the SAM 3D project (MIT License).

