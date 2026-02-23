#!/bin/bash
# Disable GPU worker to stop costs
# Usage: ./gpu_disable.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  GPU Worker Disable Script"
echo "=============================================="
echo ""
echo "This will stop the GPU worker instance."
echo "GPU processing will no longer be available until re-enabled."
echo ""

read -p "Do you want to proceed? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

cd "$TERRAFORM_DIR"

echo ""
echo "Disabling GPU worker..."
echo ""

terraform apply -var="gpu_enabled=false" -auto-approve

echo ""
echo "=============================================="
echo "  GPU Worker Disabled!"
echo "=============================================="
echo ""
echo "The GPU worker has been terminated."
echo "No GPU costs will be incurred."
echo ""
echo "To re-enable for demos:"
echo "  ./scripts/gpu_enable.sh"
echo ""
