#!/bin/bash
# Lameness Detection System - Deployment Script
# This script handles fresh deployments and database migrations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============== CONFIGURATION ==============
# Override these via environment variables or .env file

# Host configuration
DEPLOY_HOST="${DEPLOY_HOST:-localhost}"

# Service ports
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
NATS_PORT="${NATS_PORT:-4222}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# Database configuration
POSTGRES_USER="${POSTGRES_USER:-lameness_user}"
POSTGRES_DB="${POSTGRES_DB:-lameness_db}"

# Qdrant configuration
QDRANT_EMBEDDING_SIZE="${QDRANT_EMBEDDING_SIZE:-768}"

# Load .env file if it exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}Loading configuration from .env file...${NC}"
    export $(grep -v '^#' .env | xargs)
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Lameness Detection System Deployment ${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Host: ${DEPLOY_HOST}${NC}"
echo -e "${BLUE}========================================${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Use Docker Hub images when DOCKER_HUB_USER is set (see docker-compose.images.yml)
if [ -n "${DOCKER_HUB_USER:-}" ]; then
  export COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}:docker-compose.images.yml"
  echo -e "${BLUE}Using Docker Hub images: ${DOCKER_HUB_USER}/lameness-*${NC}"
fi

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    echo -e "${YELLOW}Waiting for $service to be ready...${NC}"
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://${DEPLOY_HOST}:$port" > /dev/null 2>&1 || \
           docker compose exec -T $service echo "ready" > /dev/null 2>&1; then
            echo -e "${GREEN}$service is ready!${NC}"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done
    echo -e "${RED}$service failed to start!${NC}"
    return 1
}

# Parse arguments
CLEAN_START=false
SKIP_BUILD=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clean) CLEAN_START=true ;;
        --skip-build) SKIP_BUILD=true ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean       Clean start (remove volumes and rebuild)"
            echo "  --skip-build  Skip Docker image building"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Guardrail: --skip-build should mean "pull + run" (no local build fallback).
if [ "$SKIP_BUILD" = true ] && [ -z "${DOCKER_HUB_USER:-}" ]; then
    echo -e "${RED}Error: --skip-build requires DOCKER_HUB_USER in .env (Docker Hub pull mode).${NC}"
    echo -e "${YELLOW}Add DOCKER_HUB_USER=<your-dockerhub-username> to .env and retry.${NC}"
    exit 1
fi

# Step 1: Clean start if requested
if [ "$CLEAN_START" = true ]; then
    echo -e "${YELLOW}Step 1: Cleaning up existing deployment...${NC}"
    docker compose down -v --remove-orphans 2>/dev/null || true
    rm -rf data/results/* 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete!${NC}"
else
    echo -e "${YELLOW}Step 1: Stopping existing services...${NC}"
    docker compose down 2>/dev/null || true
fi

# Step 2: Create required directories
echo -e "${YELLOW}Step 2: Creating data directories...${NC}"
mkdir -p data/{videos,canonical,processed,training,results,quality_reports}
mkdir -p data/results/{yolo,sam3,dinov3,tleap,tcn,transformer,gnn,graph_transformer,ml,fusion,tracking,shap,cow_predictions}

# Step 3: Build or pull images
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}Step 3: Building Docker images...${NC}"
    # Build one service at a time to avoid concurrent heavy conda/mamba layers
    # exhausting disk space during image creation.
    SERVICES="$(python3 - <<'PY'
import json, subprocess

cfg = json.loads(subprocess.check_output(
    ["docker", "compose", "config", "--format", "json"],
    text=True
))

for name, svc in sorted((cfg.get("services") or {}).items()):
    if svc.get("build"):
        print(name)
PY
)"

    if [ -z "$SERVICES" ]; then
        echo -e "${YELLOW}No buildable services found; running default build.${NC}"
        docker compose build
    else
        COUNT=0
        TOTAL=$(printf "%s\n" "$SERVICES" | wc -l | tr -d ' ')
        for svc in $SERVICES; do
            COUNT=$((COUNT + 1))
            echo -e "${BLUE}Building service ${COUNT}/${TOTAL}: ${svc}${NC}"
            docker compose build "$svc"
        done
    fi
else
    echo -e "${YELLOW}Step 3: Skipping Docker build (--skip-build)${NC}"
    if [ -n "${DOCKER_HUB_USER:-}" ]; then
        echo -e "${YELLOW}Pulling images from Docker Hub...${NC}"
        docker compose pull
    fi
fi

# Step 4: Start infrastructure services
echo -e "${YELLOW}Step 4: Starting infrastructure services (postgres, nats, qdrant)...${NC}"
docker compose up -d postgres nats qdrant

# Wait for postgres
sleep 5
wait_for_service "postgres" "${POSTGRES_PORT}" 30

# Step 5: Initialize database
echo -e "${YELLOW}Step 5: Initializing database...${NC}"
docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" < scripts/init_db.sql 2>/dev/null || \
    docker exec -i $(docker compose ps -q postgres) psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" < scripts/init_db.sql

echo -e "${GREEN}Database initialized!${NC}"

# Step 6: Initialize Qdrant collections
echo -e "${YELLOW}Step 6: Initializing Qdrant collections...${NC}"
sleep 3

QDRANT_URL="http://${DEPLOY_HOST}:${QDRANT_PORT}"

# Create cow_embeddings collection for Re-ID
curl -s -X PUT "${QDRANT_URL}/collections/cow_embeddings" \
    -H "Content-Type: application/json" \
    -d "{
        \"vectors\": {
            \"size\": ${QDRANT_EMBEDDING_SIZE},
            \"distance\": \"Cosine\"
        }
    }" > /dev/null 2>&1 || echo "Collection may already exist"

# Create video_embeddings collection for similarity search
curl -s -X PUT "${QDRANT_URL}/collections/video_embeddings" \
    -H "Content-Type: application/json" \
    -d "{
        \"vectors\": {
            \"size\": ${QDRANT_EMBEDDING_SIZE},
            \"distance\": \"Cosine\"
        }
    }" > /dev/null 2>&1 || echo "Collection may already exist"

echo -e "${GREEN}Qdrant collections initialized!${NC}"

# Step 7: Start all services
echo -e "${YELLOW}Step 7: Starting all services...${NC}"
if [ "$SKIP_BUILD" = true ]; then
    # Strict no-build mode: fail fast if an image is missing instead of building locally.
    docker compose up -d --no-build
else
    docker compose up -d
fi

# Step 8: Wait for key services
echo -e "${YELLOW}Step 8: Waiting for services to be healthy...${NC}"
sleep 10

wait_for_service "admin-backend" "${BACKEND_PORT}" 60
wait_for_service "admin-frontend" "${FRONTEND_PORT}" 30

# Step 9: Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!                 ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Services available at:"
echo -e "  ${BLUE}Frontend:${NC}      http://${DEPLOY_HOST}:${FRONTEND_PORT}"
echo -e "  ${BLUE}Backend API:${NC}   http://${DEPLOY_HOST}:${BACKEND_PORT}"
echo -e "  ${BLUE}API Docs:${NC}      http://${DEPLOY_HOST}:${BACKEND_PORT}/docs"
echo -e "  ${BLUE}NATS:${NC}          ${DEPLOY_HOST}:${NATS_PORT}"
echo -e "  ${BLUE}Qdrant:${NC}        http://${DEPLOY_HOST}:${QDRANT_PORT}"
echo -e "  ${BLUE}PostgreSQL:${NC}    ${DEPLOY_HOST}:${POSTGRES_PORT}"
echo ""
echo -e "Default admin credentials:"
echo -e "  ${YELLOW}Email:${NC}    admin@example.com"
echo -e "  ${YELLOW}Password:${NC} adminpass123"
echo ""
echo -e "To view logs: ${BLUE}docker compose logs -f${NC}"
echo -e "To stop:      ${BLUE}docker compose down${NC}"

