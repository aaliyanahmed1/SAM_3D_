# SAM 3D Production Docker Image

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY sam3d/ ./sam3d/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip3 install -e .

# Create directories
RUN mkdir -p /app/checkpoints /app/data /app/outputs /app/logs

# Download model checkpoints (optional, can be mounted)
# RUN wget -P /app/checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sam3d; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python3", "-m", "sam3d.api.server"]

