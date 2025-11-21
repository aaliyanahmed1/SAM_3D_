# SAM 3D - Quick Start Guide

Get started with SAM 3D in 5 minutes!

## 🚀 Installation (1 minute)

```bash
# Clone repository
git clone https://github.com/yourusername/SAM_3D.git
cd SAM_3D

# Install
pip install -r requirements.txt
pip install -e .
```

## 📥 Download Models (2 minutes)

```bash
# Download SAM checkpoint
bash scripts/download_models.sh

# Or manually:
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

## 🎯 First Segmentation (2 minutes)

```python
from sam3d import SAM3DSegmentor

# Initialize
segmentor = SAM3DSegmentor(model_type='vit_h', device='cuda')
segmentor.load_model('checkpoints/sam_vit_h_4b8939.pth')

# Segment with a point
masks, scores, image = segmentor.segment_with_points(
    image='your_image.jpg',
    points=[[300, 200]],  # Click on object
    labels=[1]             # 1 = foreground
)

print(f"✅ Generated {len(masks)} masks!")
```

## 📊 View Results

```python
from sam3d.utils import visualize_segmentation

visualize_segmentation(
    image=image,
    masks=masks,
    scores=scores,
    save_path='output.png'
)
```

## 🎥 Video Tracking

```python
from sam3d.segmentation import VideoSegmentor

video_segmentor = VideoSegmentor(segmentor=segmentor)

masks = video_segmentor.segment_video(
    video_path='video.mp4',
    initial_prompt={'points': [[100, 100]], 'labels': [1]},
    output_path='tracked.mp4'
)
```

## 🏗️ 3D Reconstruction

```python
from sam3d import Object3DReconstructor

reconstructor = Object3DReconstructor()

result = reconstructor.reconstruct_from_image('image.jpg')
reconstructor.save_mesh(result['mesh'], 'model.obj')
```

## 🐳 Docker (Alternative)

```bash
# Run with Docker
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  yourusername/sam3d:latest
```

## 📚 Next Steps

- [Examples](examples/README.md) - More detailed examples
- [API Reference](docs/api_reference.md) - Complete API docs
- [Use Cases](docs/use_cases.md) - Real-world applications
- [Contributing](CONTRIBUTING.md) - Contribute to SAM 3D

## 🆘 Troubleshooting

### CUDA Out of Memory

```python
# Use smaller model
segmentor = SAM3DSegmentor(model_type='vit_b')  # Instead of vit_h

# Or use CPU
segmentor = SAM3DSegmentor(device='cpu')
```

### Import Errors

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Model Not Found

```bash
# Check checkpoint path
ls -lh checkpoints/

# Re-download
bash scripts/download_models.sh
```

## 💬 Get Help

- 📖 [Full Documentation](docs/)
- 💡 [GitHub Discussions](https://github.com/yourusername/SAM_3D/discussions)
- 🐛 [Report Issues](https://github.com/yourusername/SAM_3D/issues)

---

**Ready to build amazing computer vision applications? Let's go! 🚀**

