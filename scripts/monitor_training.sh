#!/bin/bash
# Training Monitor Script
# Checks training every 10 minutes and restarts if crashed

PROJECT_DIR="/Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless"
LOG_FILE="$PROJECT_DIR/training_monitor.log"
TERMINAL_FILE="/Users/mehmetimga/.cursor/projects/Users-mehmetimga-ai-campions-vision-sam3-yolo-lameless/terminals/5.txt"

echo "$(date): Training monitor started" >> "$LOG_FILE"

check_training() {
    # Check if yolo process is running
    if pgrep -f "yolo pose train" > /dev/null; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

get_current_epoch() {
    tail -50 "$TERMINAL_FILE" 2>/dev/null | grep -oE '[0-9]+/150' | tail -1 | cut -d'/' -f1
}

restart_training() {
    echo "$(date): Restarting training..." >> "$LOG_FILE"
    cd "$PROJECT_DIR"
    
    # Check if we have a last.pt checkpoint
    if [ -f "runs/pose/cow_pose_full/weights/last.pt" ]; then
        echo "$(date): Resuming from checkpoint" >> "$LOG_FILE"
        nohup yolo pose train resume=True model=runs/pose/cow_pose_full/weights/last.pt > /tmp/yolo_training.log 2>&1 &
    else
        echo "$(date): Starting fresh training" >> "$LOG_FILE"
        nohup yolo pose train \
            data=data/cow_pose_full/cow_pose_full.yaml \
            model=yolov8m-pose.pt \
            epochs=150 \
            imgsz=640 \
            batch=16 \
            device=mps \
            workers=4 \
            patience=30 \
            project=runs/pose \
            name=cow_pose_full \
            exist_ok=True > /tmp/yolo_training.log 2>&1 &
    fi
    
    sleep 10
}

# Main monitoring loop
while true; do
    EPOCH=$(get_current_epoch)
    
    if check_training; then
        echo "$(date): Training running - Epoch $EPOCH/150" >> "$LOG_FILE"
    else
        # Check if training completed
        if [ -f "$PROJECT_DIR/runs/pose/cow_pose_full/weights/best.pt" ]; then
            BEST_SIZE=$(stat -f%z "$PROJECT_DIR/runs/pose/cow_pose_full/weights/best.pt" 2>/dev/null || echo 0)
            if [ "$BEST_SIZE" -gt 1000000 ]; then
                # Check results.csv for completion
                LAST_EPOCH=$(tail -1 "$PROJECT_DIR/runs/pose/cow_pose_full/results.csv" 2>/dev/null | cut -d',' -f1 | tr -d ' ')
                if [ "$LAST_EPOCH" -ge 140 ]; then
                    echo "$(date): Training COMPLETED! Last epoch: $LAST_EPOCH" >> "$LOG_FILE"
                    echo "TRAINING COMPLETE" > "$PROJECT_DIR/training_complete.flag"
                    exit 0
                fi
            fi
        fi
        
        echo "$(date): Training stopped at epoch $EPOCH - Restarting..." >> "$LOG_FILE"
        restart_training
    fi
    
    # Wait 10 minutes
    sleep 600
done


