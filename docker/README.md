# SAM 3D Docker Deployment

Production-ready Docker containers for SAM 3D.

## 🐳 Quick Start

### Build Image

```bash
# From project root
docker build -t sam3d:latest -f docker/Dockerfile .
```

### Run Container

```bash
# Run API server
docker run --gpus all -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints sam3d:latest

# Run with custom configuration
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -e SAM3D_MODEL_TYPE=vit_h \
  -e SAM3D_DEVICE=cuda \
  sam3d:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## 📋 Services

### sam3d-api

Main API server for HTTP requests.

- **Port**: 8000
- **Endpoint**: http://localhost:8000
- **GPU**: Required

### sam3d-worker

Background worker for batch processing.

- **Purpose**: Process queued jobs
- **GPU**: Required

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3D_MODEL_TYPE` | `vit_h` | Model variant (vit_h, vit_l, vit_b) |
| `SAM3D_DEVICE` | `cuda` | Device (cuda, cpu) |
| `SAM3D_LOG_LEVEL` | `INFO` | Logging level |
| `SAM3D_CHECKPOINT_PATH` | `/app/checkpoints/sam_vit_h_4b8939.pth` | Model checkpoint |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device IDs |

### Volumes

| Container Path | Purpose | Mode |
|---------------|---------|------|
| `/app/checkpoints` | Model checkpoints | Read-only |
| `/app/data` | Input data | Read-write |
| `/app/outputs` | Output files | Read-write |
| `/app/logs` | Application logs | Read-write |

## 🚀 Production Deployment

### 1. Build Production Image

```bash
docker build -t sam3d:1.0.0 -f docker/Dockerfile .
docker tag sam3d:1.0.0 sam3d:latest
```

### 2. Push to Registry

```bash
# Docker Hub
docker tag sam3d:latest yourusername/sam3d:latest
docker push yourusername/sam3d:latest

# AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag sam3d:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/sam3d:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/sam3d:latest
```

### 3. Deploy to Kubernetes

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

### 4. Deploy to Cloud Run (GCP)

```bash
gcloud run deploy sam3d \
  --image gcr.io/project-id/sam3d:latest \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-tesla-t4
```

## 🔍 Monitoring

### Health Check

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' sam3d-api

# Manual health check
curl http://localhost:8000/health
```

### Logs

```bash
# View logs
docker logs sam3d-api -f

# With Docker Compose
docker-compose logs -f sam3d-api
```

### Resource Usage

```bash
# Monitor container stats
docker stats sam3d-api

# With Docker Compose
docker-compose stats
```

## 🐛 Troubleshooting

### Issue: CUDA not available

```bash
# Check GPU access
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Issue: Out of memory

```bash
# Limit memory
docker run --gpus all --memory=8g -p 8000:8000 sam3d:latest

# Use smaller model
docker run --gpus all -e SAM3D_MODEL_TYPE=vit_b sam3d:latest
```

### Issue: Slow startup

```bash
# Download model before running
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Mount checkpoint directory
docker run --gpus all -v $(pwd)/checkpoints:/app/checkpoints sam3d:latest
```

## 🔐 Security

### Best Practices

1. **Don't run as root**
   ```dockerfile
   RUN useradd -m -u 1000 sam3d
   USER sam3d
   ```

2. **Limit capabilities**
   ```yaml
   security_opt:
     - no-new-privileges:true
   cap_drop:
     - ALL
   ```

3. **Use secrets for API keys**
   ```bash
   docker secret create openai_key openai_key.txt
   ```

4. **Scan for vulnerabilities**
   ```bash
   docker scan sam3d:latest
   ```

## 📊 Performance Optimization

### Multi-stage Build

```dockerfile
# Build stage
FROM python:3.10 as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
```

### Layer Caching

```bash
# Order Dockerfile commands by change frequency
# 1. System dependencies (rarely change)
# 2. Requirements (change occasionally)
# 3. Application code (change frequently)
```

### Image Size

```bash
# Check image size
docker images sam3d

# Analyze layers
docker history sam3d:latest

# Use alpine base (smaller)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-alpine
```

## 🔄 CI/CD Integration

### GitHub Actions

See `.github/workflows/deploy.yml` for automated builds.

### GitLab CI

```yaml
docker-build:
  stage: build
  script:
    - docker build -t sam3d:$CI_COMMIT_SHA .
    - docker push sam3d:$CI_COMMIT_SHA
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Kubernetes](https://kubernetes.io/)

