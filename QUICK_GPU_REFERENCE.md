# Quick GPU Reference Card

## 🚀 Quick Commands

### Local Development

```bash
# Build all GPU images
docker compose -f docker-compose.gpu.yml build

# Start GPU services
docker compose -f docker-compose.gpu.yml up -d

# View logs
docker compose -f docker-compose.gpu.yml logs -f yolo-pipeline

# Stop services
docker compose -f docker-compose.gpu.yml down

# Check GPU usage
watch -n 1 nvidia-smi
```

### Build and Deploy

```bash
# Build all GPU images locally
./scripts/build-gpu-images.sh

# Build specific service
./scripts/build-gpu-images.sh --service=yolo-pipeline

# Build and push to ECR
export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
./scripts/build-gpu-images.sh --push --tag=v1.0.0

# Build without cache
./scripts/build-gpu-images.sh --no-cache
```

### AWS GPU Worker

```bash
# Start GPU worker
./scripts/gpu-worker.sh start

# Check status
./scripts/gpu-worker.sh status

# View logs
./scripts/gpu-worker.sh logs

# Stop GPU worker (IMPORTANT: Stop to save costs!)
./scripts/gpu-worker.sh stop
```

## 📦 GPU Services

| Service | Dockerfile | Purpose |
|---------|------------|---------|
| yolo-pipeline | `services/yolo-pipeline/Dockerfile.gpu` | Object detection |
| sam3-pipeline | `services/sam3-pipeline/Dockerfile.gpu` | Segmentation |
| dinov3-pipeline | `services/dinov3-pipeline/Dockerfile.gpu` | Feature extraction |
| tleap-pipeline | `services/tleap-pipeline/Dockerfile.gpu` | Pose estimation |
| tcn-pipeline | `services/tcn-pipeline/Dockerfile.gpu` | Temporal CNN |
| transformer-pipeline | `services/transformer-pipeline/Dockerfile.gpu` | Gait transformer |
| gnn-pipeline | `services/gnn-pipeline/Dockerfile.gpu` | Graph neural network |
| graph-transformer-pipeline | `services/graph-transformer-pipeline/Dockerfile.gpu` | Graph transformer |

## 🔍 Verification

```bash
# Verify CUDA in container
docker run --rm --gpus all yolo-pipeline-gpu:latest \
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Test GPU access in running container
docker compose -f docker-compose.gpu.yml exec yolo-pipeline nvidia-smi
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not detected | Run `nvidia-smi`, restart Docker daemon |
| OOM errors | Reduce batch size, use larger GPU |
| Build failure | Use `--no-cache`, check network |
| ECR push fails | Re-login: `aws ecr get-login-password \| docker login` |
| Slow inference | Check GPU util with `nvidia-smi`, verify CUDA |

## 📚 Documentation

- **Full Setup Guide**: [docs/GPU_SETUP.md](docs/GPU_SETUP.md)
- **Implementation Summary**: [GPU_BUILD_SUMMARY.md](GPU_BUILD_SUMMARY.md)
- **Main README**: [README.md](README.md)

## 💰 Cost Reminders

- **g4dn.xlarge Spot**: ~$0.16/hour (70% savings)
- **g4dn.xlarge On-Demand**: ~$0.53/hour
- **Always stop GPU workers when not in use!**

## 🔧 Environment Variables

```bash
# For ECR push
export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
export AWS_REGION=us-east-1

# For GPU runtime
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES=0
```

## 📊 Monitoring

```bash
# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1

# Docker stats
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Container logs
docker compose -f docker-compose.gpu.yml logs -f --tail=100
```

---

**Quick Tip**: Keep this file open while working with GPU services!
