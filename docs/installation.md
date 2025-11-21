## Installation Guide

Complete installation instructions for SAM 3D.

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Storage**: 10GB free space
- **GPU** (optional): NVIDIA GPU with 4GB+ VRAM

### Recommended Requirements

- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.7 or higher
- **Storage**: 20GB free space (for models and data)

## Installation Methods

### Method 1: pip Install (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install SAM 3D
pip install sam3d

# Or install from GitHub
pip install git+https://github.com/yourusername/SAM_3D.git
```

### Method 2: From Source

```bash
# Clone repository
git clone https://github.com/yourusername/SAM_3D.git
cd SAM_3D

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Method 3: Docker

```bash
# Pull image
docker pull yourusername/sam3d:latest

# Or build from source
git clone https://github.com/yourusername/SAM_3D.git
cd SAM_3D
docker build -t sam3d:latest -f docker/Dockerfile .

# Run container
docker run --gpus all -p 8000:8000 sam3d:latest
```

## Download Model Checkpoints

SAM 3D requires model checkpoints to function. Choose one:

### ViT-H (Best Quality, Recommended)

```bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

**Size**: 2.4GB | **Params**: 636M | **Use**: Production, high-quality

### ViT-L (Balanced)

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P checkpoints/
```

**Size**: 1.2GB | **Params**: 308M | **Use**: Balanced performance

### ViT-B (Fastest)

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

**Size**: 375MB | **Params**: 91M | **Use**: Real-time, resource-constrained

## Verify Installation

```python
# Test import
import sam3d
print(f"SAM 3D version: {sam3d.__version__}")

# Test initialization
from sam3d import SAM3DSegmentor
segmentor = SAM3DSegmentor(model_type='vit_h', device='cpu')
print("✅ Installation successful!")
```

## GPU Setup

### NVIDIA CUDA

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Platform-Specific Instructions

### Windows

```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install SAM 3D
pip install sam3d

# If using GPU, install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install SAM 3D
pip3 install sam3d

# Note: GPU acceleration not available on macOS
```

### Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.10 python3-pip python3-venv ffmpeg libsm6 libxext6

# Install SAM 3D
pip3 install sam3d

# For GPU support, install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda
```

## Troubleshooting

### Issue: Import Error

```bash
# Solution: Reinstall
pip uninstall sam3d
pip install sam3d --no-cache-dir
```

### Issue: CUDA Out of Memory

```python
# Solution 1: Use smaller model
segmentor = SAM3DSegmentor(model_type='vit_b')  # Instead of vit_h

# Solution 2: Use CPU
segmentor = SAM3DSegmentor(device='cpu')

# Solution 3: Reduce batch size
config.batch_size = 1
```

### Issue: Slow Performance

```bash
# Check GPU usage
nvidia-smi

# Optimize model
segmentor.loader.optimize_for_inference()

# Use FP16
config.model.precision = 'fp16'
```

## Optional Dependencies

### For Advanced 3D Features (PyTorch3D)

PyTorch3D requires special installation:

```bash
# Option 1: From source (recommended)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Option 2: Using conda
conda install pytorch3d -c pytorch3d

# Note: PyTorch3D is optional and only needed for advanced 3D reconstruction
```

### For 3D Visualization

```bash
pip install open3d
```

### For Video Processing

```bash
pip install moviepy ffmpeg-python
```

### For Human Pose

```bash
pip install mediapipe
```

### All Optional Dependencies

```bash
pip install -r requirements-optional.txt
```

### For Development

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Next Steps

After installation:

1. ✅ Download model checkpoints
2. ✅ Verify installation
3. ✅ Run examples: `python examples/basic_segmentation.py`
4. ✅ Read [API Documentation](api_reference.md)
5. ✅ Check [Use Cases](use_cases.md)

## Getting Help

- 📚 [Documentation](https://sam3d.readthedocs.io)
- 💬 [Discussions](https://github.com/yourusername/SAM_3D/discussions)
- 🐛 [Issues](https://github.com/yourusername/SAM_3D/issues)
- 📧 Email: support@sam3d.com

