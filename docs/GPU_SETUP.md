# GPU Setup and Build Guide

This guide explains how to build, deploy, and use GPU-enabled Docker images for the ML pipeline services.

## Table of Contents

- [Prerequisites](#prerequisites)
- [GPU Services](#gpu-services)
- [Local Development](#local-development)
- [Building GPU Images](#building-gpu-images)
- [AWS Deployment](#aws-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- Minimum 8GB GPU memory (16GB recommended for all services)
- For local development: Ubuntu 20.04+ or compatible Linux distribution

### Software Requirements

#### For Local Development

1. **NVIDIA Driver** (version 525.60.13 or later)
   ```bash
   # Check driver version
   nvidia-smi
   ```

2. **Docker** (version 20.10 or later)
   ```bash
   docker --version
   ```

3. **NVIDIA Container Toolkit**
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Verify GPU Access**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

#### For AWS Deployment

- AWS CLI configured with appropriate credentials
- Access to ECR (Elastic Container Registry)
- Terraform 1.6 or later (for infrastructure deployment)

## GPU Services

The following services have GPU-enabled variants:

| Service | Purpose | GPU Memory | Dockerfile |
|---------|---------|------------|------------|
| `yolo-pipeline` | Object detection | 4-6 GB | `services/yolo-pipeline/Dockerfile.gpu` |
| `sam3-pipeline` | Segmentation | 6-8 GB | `services/sam3-pipeline/Dockerfile.gpu` |
| `dinov3-pipeline` | Feature extraction | 4-6 GB | `services/dinov3-pipeline/Dockerfile.gpu` |
| `tleap-pipeline` | Pose estimation | 4-6 GB | `services/tleap-pipeline/Dockerfile.gpu` |
| `tcn-pipeline` | Temporal modeling | 2-4 GB | `services/tcn-pipeline/Dockerfile.gpu` |
| `transformer-pipeline` | Gait analysis | 3-5 GB | `services/transformer-pipeline/Dockerfile.gpu` |
| `gnn-pipeline` | Graph neural network | 4-6 GB | `services/gnn-pipeline/Dockerfile.gpu` |
| `graph-transformer-pipeline` | Graph transformer | 4-6 GB | `services/graph-transformer-pipeline/Dockerfile.gpu` |

**Total GPU Memory Required**: ~32-48 GB (if running all services simultaneously)

## Local Development

### Using docker-compose.gpu.yml

The `docker-compose.gpu.yml` file provides a complete development environment with GPU support.

1. **Build GPU images**:
   ```bash
   docker compose -f docker-compose.gpu.yml build
   ```

2. **Start all services**:
   ```bash
   docker compose -f docker-compose.gpu.yml up -d
   ```

3. **Start specific GPU services**:
   ```bash
   docker compose -f docker-compose.gpu.yml up -d yolo-pipeline sam3-pipeline dinov3-pipeline
   ```

4. **View logs**:
   ```bash
   docker compose -f docker-compose.gpu.yml logs -f yolo-pipeline
   ```

5. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

6. **Stop services**:
   ```bash
   docker compose -f docker-compose.gpu.yml down
   ```

### Running Individual Services

To run a single GPU service:

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -e NATS_URL=nats://localhost:4222 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  yolo-pipeline-gpu:latest
```

## Building GPU Images

### Using the Build Script

The `scripts/build-gpu-images.sh` script provides a convenient way to build all or specific GPU images.

#### Build All GPU Images

```bash
./scripts/build-gpu-images.sh
```

#### Build Specific Service

```bash
./scripts/build-gpu-images.sh --service=yolo-pipeline
```

#### Build and Push to ECR

```bash
export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
./scripts/build-gpu-images.sh --push --tag=v1.0.0
```

#### Build Without Cache

```bash
./scripts/build-gpu-images.sh --no-cache
```

#### All Options

```bash
./scripts/build-gpu-images.sh --help
```

Options:
- `--push`: Push images to ECR after building
- `--service=<name>`: Build specific service only
- `--tag=<tag>`: Tag for the images (default: latest)
- `--platform=<arch>`: Platform architecture (default: linux/amd64)
- `--no-cache`: Build without cache

### Manual Docker Build

To manually build a GPU image:

```bash
# Build YOLO pipeline
docker build \
  -f services/yolo-pipeline/Dockerfile.gpu \
  -t yolo-pipeline-gpu:latest \
  .

# Build with specific tag
docker build \
  -f services/sam3-pipeline/Dockerfile.gpu \
  -t sam3-pipeline-gpu:v1.0.0 \
  .
```

### Verifying GPU Support

After building, verify CUDA is properly configured:

```bash
docker run --rm --gpus all yolo-pipeline-gpu:latest \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
```

## AWS Deployment

### Building and Pushing to ECR

#### 1. Create ECR Repositories

```bash
# Set AWS region
export AWS_REGION=us-east-1

# Create repositories for each GPU service
for service in yolo-pipeline sam3-pipeline dinov3-pipeline tleap-pipeline tcn-pipeline transformer-pipeline gnn-pipeline graph-transformer-pipeline; do
  aws ecr create-repository \
    --repository-name ${service}-gpu \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 || echo "Repository ${service}-gpu already exists"
done
```

#### 2. Login to ECR

```bash
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REGISTRY
```

#### 3. Build and Push Images

```bash
export ECR_REGISTRY=$(aws ecr describe-registry --region $AWS_REGION --query 'registryId' --output text).dkr.ecr.$AWS_REGION.amazonaws.com

./scripts/build-gpu-images.sh --push --tag=latest
```

### Using GitHub Actions

The `.github/workflows/build-gpu-images.yml` workflow automatically builds and pushes GPU images.

#### Manual Trigger

1. Go to GitHub Actions tab
2. Select "Build and Push GPU Images" workflow
3. Click "Run workflow"
4. Optional: Specify a tag or specific service

#### Automatic Trigger

The workflow automatically runs when:
- Changes are pushed to `main` branch
- GPU Dockerfiles or environment files are modified

### Deploying to AWS GPU Worker

The GPU worker is managed via Terraform and auto-scaling:

#### Enable GPU Worker

```bash
# Using GitHub Actions
# Go to Actions → GPU Toggle → Run workflow → Select "enable"

# OR using AWS CLI
./scripts/gpu-worker.sh start
```

#### Check Status

```bash
./scripts/gpu-worker.sh status
```

#### View Logs

```bash
./scripts/gpu-worker.sh logs
```

#### Disable GPU Worker

```bash
./scripts/gpu-worker.sh stop
```

### GPU Worker Configuration

The GPU worker configuration is in `terraform/modules/gpu_worker/`:

- **Instance Type**: g4dn.xlarge (1x NVIDIA T4, 4 vCPUs, 16 GB RAM)
- **Fallback Types**: g4dn.2xlarge, g5.2xlarge
- **Spot Instances**: Enabled by default (70% cost savings)
- **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1

## Troubleshooting

### GPU Not Detected

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Check Docker GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. Verify container has GPU access:
   ```bash
   docker compose -f docker-compose.gpu.yml exec yolo-pipeline nvidia-smi
   ```

4. Check environment variables:
   ```bash
   echo $NVIDIA_VISIBLE_DEVICES  # Should be "all" or device ID
   echo $CUDA_VISIBLE_DEVICES    # Should be "0" or device ID
   ```

### Out of Memory (OOM) Errors

**Problem**: GPU runs out of memory

**Solutions**:
1. Reduce batch size in service configuration
2. Run fewer services simultaneously
3. Use GPU with more memory (e.g., upgrade from T4 to A10)
4. Enable gradient checkpointing (if supported by model)

### Build Failures

**Problem**: Docker build fails during PyTorch installation

**Solutions**:
1. Check network connectivity
2. Use `--no-cache` flag:
   ```bash
   ./scripts/build-gpu-images.sh --no-cache --service=yolo-pipeline
   ```

3. Verify base image compatibility:
   ```bash
   docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
   ```

### Slow Build Times

**Problem**: Building GPU images takes too long

**Solutions**:
1. Use Docker BuildKit caching:
   ```bash
   DOCKER_BUILDKIT=1 docker build ...
   ```

2. Build specific services only:
   ```bash
   ./scripts/build-gpu-images.sh --service=yolo-pipeline
   ```

3. Use GitHub Actions with caching (automatically enabled)

### ECR Push Failures

**Problem**: Cannot push images to ECR

**Solutions**:
1. Verify AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

2. Re-authenticate with ECR:
   ```bash
   aws ecr get-login-password --region $AWS_REGION | \
     docker login --username AWS --password-stdin $ECR_REGISTRY
   ```

3. Check IAM permissions (requires `ecr:*` permissions)

### Performance Issues

**Problem**: GPU services running slower than expected

**Solutions**:
1. Check GPU utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. Verify CUDA version compatibility:
   ```bash
   docker exec yolo-pipeline python -c "import torch; print(torch.version.cuda)"
   ```

3. Check for CPU bottlenecks (data loading, preprocessing)
4. Ensure using CUDA-optimized operations

## Best Practices

1. **Resource Management**
   - Always stop GPU workers when not in use to save costs
   - Monitor GPU memory usage with `nvidia-smi`
   - Use appropriate batch sizes for your GPU

2. **Development Workflow**
   - Build and test locally with `docker-compose.gpu.yml`
   - Push to ECR only when ready for production
   - Use versioned tags (not just `latest`) for production

3. **Cost Optimization**
   - Use spot instances (default configuration)
   - Scale down GPU workers when idle
   - Consider using smaller instance types for development

4. **Image Management**
   - Tag images with meaningful versions
   - Clean up old images regularly
   - Use ECR lifecycle policies to auto-delete old images

5. **Monitoring**
   - Set up CloudWatch alarms for GPU utilization
   - Monitor service logs for CUDA errors
   - Track GPU memory usage trends

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
