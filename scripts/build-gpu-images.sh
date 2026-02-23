#!/bin/bash
# Build script for GPU-enabled Docker images
# Usage: ./scripts/build-gpu-images.sh [OPTIONS]
#
# Options:
#   --push              Push images to ECR after building
#   --service=<name>    Build specific service only
#   --tag=<tag>         Tag for the images (default: latest)
#   --platform=<arch>   Platform architecture (default: linux/amd64)
#   --no-cache          Build without cache
#   --help              Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PUSH=false
TAG="latest"
PLATFORM="linux/amd64"
USE_CACHE=true
SPECIFIC_SERVICE=""
ECR_REGISTRY="${ECR_REGISTRY:-}"

# GPU services list
GPU_SERVICES=(
    "yolo-pipeline"
    "sam3-pipeline"
    "dinov3-pipeline"
    "tleap-pipeline"
    "tcn-pipeline"
    "transformer-pipeline"
    "gnn-pipeline"
    "graph-transformer-pipeline"
)

print_header() {
    echo ""
    echo -e "${BLUE}=============================================="
    echo "  GPU Image Builder"
    echo -e "==============================================${NC}"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build GPU-enabled Docker images for ML pipelines

Options:
    --push                  Push images to ECR after building
    --service=<name>        Build specific service only
    --tag=<tag>            Tag for the images (default: latest)
    --platform=<arch>      Platform architecture (default: linux/amd64)
    --no-cache             Build without cache
    --help                 Show this help message

Examples:
    # Build all GPU images
    $0

    # Build specific service
    $0 --service=yolo-pipeline

    # Build and push to ECR
    $0 --push --tag=v1.0.0

    # Build without cache
    $0 --no-cache

Environment Variables:
    ECR_REGISTRY           ECR registry URL (required for --push)

GPU Services:
    ${GPU_SERVICES[@]}

EOF
}

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --push)
            PUSH=true
            shift
            ;;
        --service=*)
            SPECIFIC_SERVICE="${arg#*=}"
            shift
            ;;
        --tag=*)
            TAG="${arg#*=}"
            shift
            ;;
        --platform=*)
            PLATFORM="${arg#*=}"
            shift
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate ECR_REGISTRY if push is requested
if [ "$PUSH" = true ] && [ -z "$ECR_REGISTRY" ]; then
    echo -e "${RED}ERROR: ECR_REGISTRY environment variable must be set when using --push${NC}"
    echo "Example: export ECR_REGISTRY=123456789012.dkr.ecr.us-west-2.amazonaws.com"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    exit 1
fi

# Check if Docker Buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo -e "${YELLOW}WARNING: Docker Buildx not found, using standard docker build${NC}"
    USE_BUILDX=false
else
    USE_BUILDX=true
fi

build_service() {
    local service=$1
    local dockerfile="services/${service}/Dockerfile.gpu"

    if [ ! -f "$dockerfile" ]; then
        echo -e "${RED}ERROR: GPU Dockerfile not found for ${service}: ${dockerfile}${NC}"
        return 1
    fi

    local image_name="${service}-gpu"
    if [ -n "$ECR_REGISTRY" ]; then
        image_name="${ECR_REGISTRY}/${service}-gpu"
    fi

    echo -e "${BLUE}Building ${service} GPU image...${NC}"
    echo "  Dockerfile: ${dockerfile}"
    echo "  Image: ${image_name}:${TAG}"
    echo "  Platform: ${PLATFORM}"

    local build_args=""
    if [ "$USE_CACHE" = false ]; then
        build_args="$build_args --no-cache"
    fi

    local start_time=$(date +%s)

    if [ "$USE_BUILDX" = true ]; then
        docker buildx build \
            --platform="${PLATFORM}" \
            --file="${dockerfile}" \
            --tag="${image_name}:${TAG}" \
            --tag="${image_name}:latest" \
            ${build_args} \
            --progress=plain \
            .
    else
        docker build \
            --file="${dockerfile}" \
            --tag="${image_name}:${TAG}" \
            --tag="${image_name}:latest" \
            ${build_args} \
            .
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo -e "${GREEN}✓ Successfully built ${service} GPU image (${duration}s)${NC}"
    echo ""

    if [ "$PUSH" = true ]; then
        echo -e "${BLUE}Pushing ${service} GPU image to ECR...${NC}"
        docker push "${image_name}:${TAG}"
        docker push "${image_name}:latest"
        echo -e "${GREEN}✓ Successfully pushed ${service} GPU image${NC}"
        echo ""
    fi
}

login_to_ecr() {
    if [ -z "$ECR_REGISTRY" ]; then
        return
    fi

    echo -e "${BLUE}Logging in to Amazon ECR...${NC}"

    # Extract region from ECR registry URL
    local region=$(echo "$ECR_REGISTRY" | cut -d'.' -f4)

    aws ecr get-login-password --region "$region" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"

    echo -e "${GREEN}✓ Successfully logged in to ECR${NC}"
    echo ""
}

# Main execution
print_header

echo "Configuration:"
echo "  Tag: ${TAG}"
echo "  Platform: ${PLATFORM}"
echo "  Use Cache: ${USE_CACHE}"
echo "  Push to ECR: ${PUSH}"
if [ -n "$SPECIFIC_SERVICE" ]; then
    echo "  Service: ${SPECIFIC_SERVICE}"
else
    echo "  Services: All GPU services (${#GPU_SERVICES[@]})"
fi
echo ""

# Login to ECR if push is requested
if [ "$PUSH" = true ]; then
    login_to_ecr
fi

# Build images
total_start=$(date +%s)

if [ -n "$SPECIFIC_SERVICE" ]; then
    # Build specific service
    if [[ " ${GPU_SERVICES[@]} " =~ " ${SPECIFIC_SERVICE} " ]]; then
        build_service "$SPECIFIC_SERVICE"
    else
        echo -e "${RED}ERROR: Unknown service: ${SPECIFIC_SERVICE}${NC}"
        echo "Available GPU services: ${GPU_SERVICES[@]}"
        exit 1
    fi
else
    # Build all services
    successful=0
    failed=0

    for service in "${GPU_SERVICES[@]}"; do
        if build_service "$service"; then
            ((successful++))
        else
            ((failed++))
            echo -e "${RED}✗ Failed to build ${service}${NC}"
            echo ""
        fi
    done

    total_end=$(date +%s)
    total_duration=$((total_end - total_start))

    echo -e "${BLUE}=============================================="
    echo "  Build Summary"
    echo -e "==============================================${NC}"
    echo ""
    echo -e "${GREEN}Successful: ${successful}${NC}"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}Failed: ${failed}${NC}"
    fi
    echo "Total time: ${total_duration}s"
    echo ""

    if [ $failed -gt 0 ]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ All GPU images built successfully!${NC}"
echo ""

if [ "$PUSH" = false ]; then
    echo "To push images to ECR, run with --push flag"
    echo "Example: $0 --push --tag=${TAG}"
    echo ""
fi
