#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SERVICES=("$@")

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU=true
  PROFILE="gpu"
  VIDEO_SERVICE="video-preprocessing"
else
  GPU=false
  PROFILE="cpu"
  VIDEO_SERVICE="video-preprocessing-cpu"
fi

export COMPOSE_PROFILES="${PROFILE}"
COMPOSE_FILE="docker-compose.yml"

RESOLVED_SERVICES=()
for service in "${SERVICES[@]}"; do
  if [[ "$service" == "video-preprocessing" ]]; then
    RESOLVED_SERVICES+=("${VIDEO_SERVICE}")
  else
    RESOLVED_SERVICES+=("$service")
  fi
done

echo "Using ${COMPOSE_FILE} (GPU=${GPU}, profile=${PROFILE})"

if [ ${#SERVICES[@]} -eq 0 ]; then
  docker compose -f "$COMPOSE_FILE" up --build
else
  docker compose -f "$COMPOSE_FILE" up --build "${RESOLVED_SERVICES[@]}"
fi
