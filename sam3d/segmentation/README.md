# SAM 3D Segmentation Module

This module provides comprehensive image and video segmentation capabilities using the Segment Anything Model (SAM).

## Components

### Image Segmentor (`image_segmentor.py`)

High-level interface for image segmentation with multiple prompting methods.

#### Features

- **Point prompts**: Click to select foreground/background
- **Bounding box prompts**: Draw box around object
- **Text prompts**: Describe object in natural language (SAM 3)
- **Automatic segmentation**: Segment everything in image
- **Mask refinement**: Improve existing segmentations
- **Batch processing**: Process multiple images efficiently

#### Usage

```python
from sam3d.segmentation import SAM3DSegmentor

# Initialize
segmentor = SAM3DSegmentor(model_type='vit_h', device='cuda')
segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')

# Point-based segmentation
masks, scores, image = segmentor.segment_with_points(
    image='photo.jpg',
    points=[[300, 200], [400, 250]],  # foreground points
    labels=[1, 1]  # 1=foreground, 0=background
)

# Box-based segmentation
masks, scores, image = segmentor.segment_with_box(
    image='photo.jpg',
    box=[100, 100, 500, 500]  # [x1, y1, x2, y2]
)

# Text-based segmentation (SAM 3 only)
masks, scores, image = segmentor.segment_with_text(
    image='photo.jpg',
    text_prompt='red car'
)

# Automatic segmentation (segment everything)
masks_list = segmentor.segment_everything(
    image='photo.jpg',
    points_per_side=32,
    pred_iou_thresh=0.88
)

# Refine existing mask
refined_masks, scores = segmentor.refine_mask(
    image='photo.jpg',
    mask=existing_mask,
    points=[[350, 225]],
    labels=[1]
)

# Batch processing
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
prompts = [
    {'points': [[100, 100]], 'labels': [1]},
    {'box': [50, 50, 200, 200]},
    {'text': 'blue bottle'}
]
results = segmentor.batch_segment(images, prompts, batch_size=2)
```

### Video Segmentor (`video_segmentor.py`)

Video object segmentation with temporal consistency.

#### Features

- **Temporal propagation**: Track objects across frames
- **Multi-object tracking**: Segment multiple objects simultaneously
- **Confidence-based re-segmentation**: Handle occlusions
- **Frame extraction**: Extract frames from videos
- **Video generation**: Create videos from frame sequences

#### Usage

```python
from sam3d.segmentation import VideoSegmentor
from sam3d.segmentation import SAM3DSegmentor

# Initialize
image_segmentor = SAM3DSegmentor(model_type='vit_h')
image_segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')

video_segmentor = VideoSegmentor(
    segmentor=image_segmentor,
    propagation_window=8,
    confidence_threshold=0.5
)

# Segment object in video
masks = video_segmentor.segment_video(
    video_path='video.mp4',
    initial_prompt={'points': [[100, 100]], 'labels': [1]},
    output_path='segmented_video.mp4',
    max_frames=None,
    show_progress=True
)

# Extract frames
frame_paths = video_segmentor.extract_frames(
    video_path='video.mp4',
    output_dir='frames/',
    frame_rate=10  # Extract every 10th frame
)

# Create video from frames
video_segmentor.create_video_from_frames(
    frame_paths=frame_paths,
    output_path='output.mp4',
    fps=30
)

# Track multiple objects
object_prompts = [
    {'points': [[100, 100]], 'labels': [1]},  # Object 1
    {'box': [200, 200, 400, 400]},            # Object 2
    {'text': 'person walking'}                # Object 3 (SAM 3)
]

all_masks = video_segmentor.segment_multi_object(
    video_path='video.mp4',
    object_prompts=object_prompts,
    output_path='multi_object_tracking.mp4'
)
```

## Prompting Methods

### 1. Point Prompts

Most basic prompting method. Click on object (foreground) or background.

```python
# Single foreground point
points = [[150, 200]]
labels = [1]

# Multiple points (foreground + background)
points = [[150, 200], [500, 100]]
labels = [1, 0]  # 1=foreground, 0=background
```

### 2. Bounding Box Prompts

Draw a box around the object.

```python
# Format: [x1, y1, x2, y2]
box = [100, 100, 500, 500]
```

### 3. Text Prompts (SAM 3 only)

Describe the object in natural language.

```python
text_prompt = "red baseball cap"
text_prompt = "person wearing blue shirt"
text_prompt = "car on the left side"
```

### 4. Mask Prompts

Refine an existing mask with additional prompts.

```python
# Start with initial segmentation
masks, _, _ = segmentor.segment_with_points(image, [[100, 100]], [1])

# Refine with additional points
refined_masks, scores = segmentor.refine_mask(
    image=image,
    mask=masks[0],
    points=[[150, 150]],
    labels=[1]
)
```

## Configuration

Configure segmentation behavior via `Config` object:

```python
from sam3d.core import Config

config = Config()
config.segmentation.multimask_output = True
config.segmentation.stability_score_threshold = 0.95
config.segmentation.box_nms_threshold = 0.7
config.segmentation.min_mask_region_area = 100

segmentor = SAM3DSegmentor(config=config)
```

## Performance Tips

### 1. Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| vit_b | Fast | Good | Real-time, low-resource |
| vit_l | Medium | Better | Balanced production |
| vit_h | Slow | Best | Offline, high-quality |

### 2. Optimization

```python
# Enable optimizations
segmentor.loader.optimize_for_inference()

# Use FP16 on GPU
config.model.precision = 'fp16'

# Reduce multimask output for speed
config.segmentation.multimask_output = False
```

### 3. Batch Processing

```python
# Process multiple images together
results = segmentor.batch_segment(
    images=image_list,
    prompts=prompt_list,
    batch_size=8  # Adjust based on GPU memory
)
```

### 4. Video Processing

```python
# Limit frames for faster processing
masks = video_segmentor.segment_video(
    video_path='video.mp4',
    initial_prompt=prompt,
    max_frames=300,  # Process first 300 frames
    show_progress=True
)

# Extract frames first for parallel processing
frame_paths = video_segmentor.extract_frames(
    video_path='video.mp4',
    output_dir='frames/',
    frame_rate=5  # Every 5th frame
)
```

## Error Handling

```python
try:
    masks, scores, image = segmentor.segment_with_points(
        image='photo.jpg',
        points=[[100, 100]],
        labels=[1]
    )
except FileNotFoundError:
    print("Image file not found")
except RuntimeError as e:
    print(f"Model not loaded: {e}")
except Exception as e:
    print(f"Segmentation error: {e}")
```

## Examples

See `/examples` for complete examples:
- `basic_segmentation.py` - Simple image segmentation
- `video_tracking.py` - Video object tracking
- `batch_processing.py` - Batch image processing
- `automatic_segmentation.py` - Segment everything

## API Reference

See [API Documentation](../../docs/api_reference.md) for complete reference.

