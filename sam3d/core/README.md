# SAM 3D Core Module

This module contains the core functionality for SAM 3D, including configuration management and model loading utilities.

## Components

### Config (`config.py`)

The configuration management system provides a flexible and type-safe way to configure all aspects of SAM 3D.

#### Features

- **Type-safe configuration** using Python dataclasses
- **YAML support** for easy configuration management
- **Auto-detection** of device (CUDA/CPU)
- **Modular design** with separate configs for each component

#### Usage

```python
from sam3d.core import Config

# Create default configuration
config = Config()

# Load from YAML file
config = Config.from_yaml('configs/model_config.yaml')

# Access configuration
print(config.model.model_type)  # vit_h
print(config.model.device)      # cuda or cpu

# Modify configuration
config.model.model_type = 'vit_l'
config.segmentation.multimask_output = False

# Save configuration
config.save('my_config.yaml')
```

#### Configuration Structure

```yaml
model:
  model_type: vit_h        # vit_h, vit_l, vit_b
  checkpoint_path: null
  device: auto             # auto, cuda, cpu
  precision: fp32          # fp32, fp16, bf16

segmentation:
  multimask_output: true
  stability_score_threshold: 0.95
  box_nms_threshold: 0.7
  min_mask_region_area: 0

reconstruction:
  depth_estimation_model: dpt_large
  point_cloud_density: 10000
  mesh_simplification_factor: 0.9
  texture_resolution: 1024
  enable_refinement: true

tracking:
  max_frames: null
  fps: null
  propagation_window: 8
  memory_bank_size: 16
  confidence_threshold: 0.5

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  log_level: info
  max_upload_size: 104857600  # 100MB
```

### ModelLoader (`model_loader.py`)

The ModelLoader class handles loading SAM models from various sources with automatic fallback.

#### Features

- **Multiple loading methods**: Local checkpoint, HuggingFace, auto-detection
- **Automatic optimization**: FP16, torch.compile, gradient disabling
- **Model information**: Parameters count, memory footprint
- **Error handling**: Clear error messages and instructions

#### Usage

```python
from sam3d.core import ModelLoader

# Initialize loader
loader = ModelLoader(model_type='vit_h', device='cuda')

# Load from local checkpoint
model = loader.load_from_checkpoint('checkpoints/sam_vit_h_4b8939.pth')

# Or load from HuggingFace
model = loader.load_from_huggingface()

# Or auto-detect best method
model = loader.load_auto(checkpoint_path='checkpoints/sam_vit_h_4b8939.pth')

# Optimize for inference
loader.optimize_for_inference()

# Get model information
info = loader.get_model_info()
print(f"Total parameters: {info['total_parameters']:,}")
print(f"Memory footprint: {info['memory_footprint_mb']:.2f} MB")

# Get predictor
predictor = loader.get_predictor()
```

#### Model Types

| Model Type | Parameters | Size | Speed | Accuracy |
|-----------|-----------|------|-------|----------|
| `vit_b`   | 91M       | 375MB | Fast | Good |
| `vit_l`   | 308M      | 1.2GB | Medium | Better |
| `vit_h`   | 636M      | 2.4GB | Slow | Best |

#### Loading Methods

1. **From Local Checkpoint**
   ```python
   loader.load_from_checkpoint('path/to/checkpoint.pth')
   ```

2. **From HuggingFace**
   ```python
   loader.load_from_huggingface('facebook/sam-vit-huge')
   ```

3. **Auto Detection**
   ```python
   loader.load_auto(checkpoint_path='optional/path.pth')
   ```

#### Optimization Options

```python
# Optimize for inference
loader.optimize_for_inference()

# This applies:
# - eval() mode
# - Gradient disabling
# - torch.compile (if available)
# - FP16 conversion (if on CUDA)
```

## Download Model Checkpoints

```bash
# Download script
bash scripts/download_models.sh

# Or manually:
# ViT-H (default, best quality)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (balanced)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (fastest)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Error Handling

Both Config and ModelLoader provide clear error messages:

```python
# Missing checkpoint
try:
    loader.load_from_checkpoint('nonexistent.pth')
except FileNotFoundError as e:
    print(e)  # "Checkpoint not found: nonexistent.pth"

# Missing package
try:
    loader.load_from_huggingface()
except ImportError as e:
    print(e)  # "transformers package required..."
```

## Best Practices

1. **Use YAML configurations** for reproducibility
2. **Auto-detect device** unless specific GPU required
3. **Optimize for inference** in production
4. **Check model info** before deployment
5. **Handle loading errors** gracefully

## Examples

See `/examples` folder for complete examples:
- `basic_segmentation.py` - Basic usage
- `batch_processing.py` - Production batch processing
- `custom_config.py` - Custom configuration

## API Reference

See [API Documentation](../../docs/api_reference.md) for complete reference.

