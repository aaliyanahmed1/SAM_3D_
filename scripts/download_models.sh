#!/bin/bash

# SAM 3D Model Download Script
# Downloads all required model checkpoints

set -e

echo "=" | head -c 60
echo
echo "SAM 3D - Model Download Script"
echo "=" | head -c 60
echo

# Create checkpoints directory
mkdir -p checkpoints
cd checkpoints

# Function to download with progress
download_model() {
    local url=$1
    local filename=$2
    local size=$3
    
    echo
    echo "Downloading $filename ($size)..."
    
    if [ -f "$filename" ]; then
        echo "✅ $filename already exists, skipping..."
    else
        wget --progress=bar:force:noscroll "$url" -O "$filename"
        echo "✅ Downloaded $filename"
    fi
}

# SAM Models
echo
echo "📦 Downloading SAM Models..."

# ViT-H (Best quality - Recommended)
download_model \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
    "sam_vit_h_4b8939.pth" \
    "2.4GB"

# ViT-L (Balanced)
echo
read -p "Download ViT-L model? (2.4GB) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_model \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" \
        "sam_vit_l_0b3195.pth" \
        "1.2GB"
fi

# ViT-B (Fastest)
echo
read -p "Download ViT-B model? (375MB) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_model \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
        "sam_vit_b_01ec64.pth" \
        "375MB"
fi

cd ..

# Verify downloads
echo
echo "=" | head -c 60
echo
echo "Verifying downloads..."
echo

if [ -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
    echo "✅ ViT-H checkpoint verified"
else
    echo "❌ ViT-H checkpoint missing"
fi

if [ -f "checkpoints/sam_vit_l_0b3195.pth" ]; then
    echo "✅ ViT-L checkpoint verified"
fi

if [ -f "checkpoints/sam_vit_b_01ec64.pth" ]; then
    echo "✅ ViT-B checkpoint verified"
fi

echo
echo "=" | head -c 60
echo
echo "✅ Model download complete!"
echo
echo "Models saved to: $(pwd)/checkpoints/"
echo
echo "Next steps:"
echo "  1. Verify installation: python -c 'import sam3d; print(sam3d.__version__)'"
echo "  2. Run examples: python examples/basic_segmentation.py"
echo "  3. Read docs: cat docs/installation.md"
echo
echo "=" | head -c 60

