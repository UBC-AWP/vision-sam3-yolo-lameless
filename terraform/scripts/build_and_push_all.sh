#!/bin/bash
# Build and push all Docker images for linux/amd64 to ECR
# This is required for AWS ECS Fargate deployment

set -e

ECR_REGISTRY="703582588105.dkr.ecr.us-west-2.amazonaws.com"
PROJECT_ROOT="/Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=============================================="
echo "  Build & Push All Images for linux/amd64"
echo -e "==============================================${NC}"

# Login to ECR
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_REGISTRY

# Services to build
declare -A SERVICES=(
    ["admin-backend"]="services/admin-interface/backend/Dockerfile"
    ["admin-frontend"]="services/admin-interface/frontend/Dockerfile"
    ["video-ingestion"]="services/video-ingestion/Dockerfile"
    ["video-preprocessing"]="services/video-preprocessing/Dockerfile"
    ["ml-pipeline"]="services/ml-pipeline/Dockerfile"
    ["fusion-service"]="services/fusion-service/Dockerfile"
    ["tracking-service"]="services/tracking-service/Dockerfile"
    ["clip-curation"]="services/clip-curation/Dockerfile"
)

for service in "${!SERVICES[@]}"; do
    dockerfile="${SERVICES[$service]}"
    echo ""
    echo -e "${YELLOW}Building $service for linux/amd64...${NC}"
    docker build --platform linux/amd64 \
        -t "$ECR_REGISTRY/$service:latest" \
        -f "$PROJECT_ROOT/$dockerfile" \
        "$PROJECT_ROOT"

    echo -e "${YELLOW}Pushing $service to ECR...${NC}"
    docker push "$ECR_REGISTRY/$service:latest"

    echo -e "${GREEN}✓ $service done${NC}"
done

echo ""
echo -e "${GREEN}=============================================="
echo "  All images built and pushed!"
echo -e "==============================================${NC}"
echo ""
echo "To deploy to ECS, run:"
echo "  aws ecs update-service --cluster cow-lameness-production-cluster --service admin-backend --force-new-deployment --region us-west-2"
