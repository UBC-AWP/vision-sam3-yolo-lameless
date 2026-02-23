#!/bin/bash
# GPU Worker Control Script
# Quick on/off control for GPU worker without running full Terraform
# Usage: ./scripts/gpu-worker.sh [start|stop|status]

set -e

# Configuration
ASG_NAME="cow-lameness-production-gpu-worker-asg"
REGION="${AWS_REGION:-us-west-2}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}=============================================="
    echo "  GPU Worker Control"
    echo -e "==============================================${NC}"
    echo ""
}

print_cost_info() {
    echo -e "${YELLOW}Cost Information:${NC}"
    echo "  • g4dn.xlarge On-Demand: ~\$0.526/hour"
    echo "  • g4dn.xlarge Spot:      ~\$0.16/hour (70% savings)"
    echo ""
}

get_status() {
    local asg_info
    asg_info=$(aws autoscaling describe-auto-scaling-groups \
        --auto-scaling-group-names "$ASG_NAME" \
        --region "$REGION" \
        --query 'AutoScalingGroups[0]' \
        --output json 2>/dev/null || echo "{}")

    if [ "$asg_info" = "{}" ] || [ -z "$asg_info" ] || [ "$asg_info" = "null" ]; then
        echo -e "${RED}ERROR: Auto Scaling Group '$ASG_NAME' not found${NC}"
        echo "Make sure you have run 'terraform apply' first."
        return 1
    fi

    local desired_capacity
    local running_instances
    local instance_ids

    desired_capacity=$(echo "$asg_info" | jq -r '.DesiredCapacity // 0')
    running_instances=$(echo "$asg_info" | jq -r '.Instances | length // 0')
    instance_ids=$(echo "$asg_info" | jq -r '.Instances[].InstanceId // empty')

    echo -e "${BLUE}Auto Scaling Group:${NC} $ASG_NAME"
    echo -e "${BLUE}Region:${NC} $REGION"
    echo ""
    echo -e "${BLUE}Desired Capacity:${NC} $desired_capacity"
    echo -e "${BLUE}Running Instances:${NC} $running_instances"

    if [ "$desired_capacity" -gt 0 ] && [ "$running_instances" -gt 0 ]; then
        echo ""
        echo -e "${GREEN}Status: GPU Worker is RUNNING${NC}"
        echo ""
        echo "Instance Details:"
        for instance_id in $instance_ids; do
            local instance_info
            instance_info=$(aws ec2 describe-instances \
                --instance-ids "$instance_id" \
                --region "$REGION" \
                --query 'Reservations[0].Instances[0]' \
                --output json 2>/dev/null)

            local instance_type
            local state
            local launch_time
            local private_ip

            instance_type=$(echo "$instance_info" | jq -r '.InstanceType // "unknown"')
            state=$(echo "$instance_info" | jq -r '.State.Name // "unknown"')
            launch_time=$(echo "$instance_info" | jq -r '.LaunchTime // "unknown"')
            private_ip=$(echo "$instance_info" | jq -r '.PrivateIpAddress // "N/A"')

            echo "  • Instance ID: $instance_id"
            echo "    Type: $instance_type"
            echo "    State: $state"
            echo "    Private IP: $private_ip"
            echo "    Launch Time: $launch_time"
        done
    elif [ "$desired_capacity" -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Status: GPU Worker is STARTING...${NC}"
        echo "Please wait 2-3 minutes for the instance to be ready."
    else
        echo ""
        echo -e "${YELLOW}Status: GPU Worker is STOPPED${NC}"
    fi
}

start_gpu() {
    print_header
    print_cost_info

    echo "Starting GPU worker..."
    echo ""

    # Set desired capacity to 1
    aws autoscaling set-desired-capacity \
        --auto-scaling-group-name "$ASG_NAME" \
        --desired-capacity 1 \
        --region "$REGION"

    echo -e "${GREEN}✓ GPU worker scaling up!${NC}"
    echo ""
    echo "The GPU worker is starting. It may take 2-3 minutes to be fully ready."
    echo ""
    echo "GPU services that will run:"
    echo "  • yolo-pipeline (object detection)"
    echo "  • sam3-pipeline (segmentation)"
    echo "  • tleap-pipeline (pose estimation)"
    echo "  • dinov3-pipeline (feature extraction)"
    echo "  • tcn-pipeline (temporal analysis)"
    echo "  • transformer-pipeline"
    echo "  • gnn-pipeline"
    echo "  • graph-transformer-pipeline"
    echo ""
    echo "To check status: ./scripts/gpu-worker.sh status"
    echo "To stop:         ./scripts/gpu-worker.sh stop"
    echo ""
}

stop_gpu() {
    print_header

    echo "Stopping GPU worker..."
    echo ""

    # Set desired capacity to 0
    aws autoscaling set-desired-capacity \
        --auto-scaling-group-name "$ASG_NAME" \
        --desired-capacity 0 \
        --region "$REGION"

    echo -e "${GREEN}✓ GPU worker scaling down!${NC}"
    echo ""
    echo "The GPU worker will terminate shortly."
    echo "No GPU costs will be incurred once terminated."
    echo ""
    echo "To restart: ./scripts/gpu-worker.sh start"
    echo ""
}

show_logs() {
    echo ""
    echo -e "${BLUE}Fetching GPU worker logs...${NC}"
    echo ""

    aws logs tail "/ec2/cow-lameness-production-gpu-worker" \
        --region "$REGION" \
        --since 30m \
        --follow 2>/dev/null || echo "No logs found. GPU worker may not be running."
}

show_usage() {
    print_header
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start   - Start the GPU worker (scale up to 1)"
    echo "  stop    - Stop the GPU worker (scale down to 0)"
    echo "  status  - Show current GPU worker status"
    echo "  logs    - Stream GPU worker logs"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start GPU for ML processing"
    echo "  $0 status    # Check if GPU is running"
    echo "  $0 stop      # Stop GPU to save costs"
    echo ""
    print_cost_info
}

# Main
case "${1:-}" in
    start)
        start_gpu
        ;;
    stop)
        stop_gpu
        ;;
    status)
        print_header
        get_status
        ;;
    logs)
        show_logs
        ;;
    *)
        show_usage
        ;;
esac
