#!/bin/bash
# Build and push Docker images to ECR
# Usage: ./push_images.sh [--service SERVICE_NAME]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

AWS_REGION="us-west-2"
AWS_ACCOUNT_ID="703582588105"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Services to build (service_name:dockerfile_path)
declare -A SERVICES=(
    ["admin-backend"]="services/admin-interface/backend"
    ["admin-frontend"]="services/admin-interface/frontend"
)

# Parse arguments
SINGLE_SERVICE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --service|-s)
            SINGLE_SERVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=============================================="
echo "  Push Docker Images to ECR"
echo -e "==============================================${NC}"
echo ""
echo "ECR Registry: $ECR_REGISTRY"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Login to ECR
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
echo ""

cd "$PROJECT_ROOT"

# Build and push function
build_and_push() {
    local service=$1
    local path=$2
    local dockerfile="$path/Dockerfile"

    if [ ! -f "$dockerfile" ]; then
        echo -e "${RED}Dockerfile not found: $dockerfile${NC}"
        return 1
    fi

    echo -e "${YELLOW}Building $service...${NC}"
    docker build -t "$ECR_REGISTRY/$service:latest" -f "$dockerfile" .

    echo -e "${YELLOW}Pushing $service...${NC}"
    docker push "$ECR_REGISTRY/$service:latest"

    echo -e "${GREEN}✓ $service pushed successfully${NC}"
    echo ""
}

# Build services
if [ -n "$SINGLE_SERVICE" ]; then
    if [ -z "${SERVICES[$SINGLE_SERVICE]}" ]; then
        echo -e "${RED}Unknown service: $SINGLE_SERVICE${NC}"
        echo "Available services: ${!SERVICES[@]}"
        exit 1
    fi
    build_and_push "$SINGLE_SERVICE" "${SERVICES[$SINGLE_SERVICE]}"
else
    for service in "${!SERVICES[@]}"; do
        build_and_push "$service" "${SERVICES[$service]}"
    done
fi

echo -e "${GREEN}=============================================="
echo "  All images pushed successfully!"
echo -e "==============================================${NC}"
echo ""
echo "To update ECS services, run:"
echo "  aws ecs update-service --cluster cow-lameness-production-cluster --service admin-backend --force-new-deployment"
echo "  aws ecs update-service --cluster cow-lameness-production-cluster --service admin-frontend --force-new-deployment"
