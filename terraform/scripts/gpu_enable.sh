#!/bin/bash
# Enable GPU worker for ML processing demos
# Usage: ./gpu_enable.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  GPU Worker Enable Script"
echo "=============================================="
echo ""
echo "This will start a GPU worker instance for ML processing."
echo ""
echo "Cost information:"
echo "  - g4dn.xlarge On-Demand: ~\$0.526/hour"
echo "  - g4dn.xlarge Spot:      ~\$0.16/hour (70% savings)"
echo ""

read -p "Do you want to proceed? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

cd "$TERRAFORM_DIR"

echo ""
echo "Enabling GPU worker..."
echo ""

terraform apply -var="gpu_enabled=true" -auto-approve

echo ""
echo "=============================================="
echo "  GPU Worker Enabled!"
echo "=============================================="
echo ""
echo "The GPU worker is starting. It may take 2-3 minutes to be fully ready."
echo ""
echo "To check status:"
echo "  aws autoscaling describe-auto-scaling-groups \\"
echo "    --auto-scaling-group-names cow-lameness-production-gpu-worker-asg \\"
echo "    --query 'AutoScalingGroups[0].Instances'"
echo ""
echo "Remember to disable when done to stop costs:"
echo "  ./scripts/gpu_disable.sh"
echo ""
