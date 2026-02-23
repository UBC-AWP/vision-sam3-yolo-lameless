#!/usr/bin/env python3
"""
Analyze the training data to check for directional bias in cow orientation.
Parses YOLO pose label files to determine if cows are facing left or right.
"""

import os
from pathlib import Path
from collections import defaultdict

# Dataset paths
TRAIN_LABELS = Path("data/cow-pose-estimation.v1i.yolov8/train/labels")
VALID_LABELS = Path("data/cow-pose-estimation.v1i.yolov8/valid/labels")
TEST_LABELS = Path("data/cow-pose-estimation.v1i.yolov8/test/labels")

# Keypoint indices (20 keypoints per cow)
# 0: nose, 1: head_neck, 2: withers, 3: mid_back
# 4: hip_left, 10: hip_right (rear of cow)
# Based on the mapping, head keypoints are 0,1 and rear keypoints are 4,10

def parse_yolo_pose_label(label_file):
    """
    Parse a YOLO pose label file.
    Format: class_id cx cy w h kp0_x kp0_y kp0_v kp1_x kp1_y kp1_v ...
    Coordinates are normalized (0-1).
    """
    detections = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # First 5 values: class, cx, cy, w, h
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # Remaining values are keypoints (x, y, visibility) triplets
            keypoints = []
            kp_data = parts[5:]
            for i in range(0, len(kp_data), 3):
                if i + 2 < len(kp_data):
                    kp_x = float(kp_data[i])
                    kp_y = float(kp_data[i + 1])
                    kp_v = float(kp_data[i + 2])  # visibility
                    keypoints.append((kp_x, kp_y, kp_v))
            
            detections.append({
                'class_id': class_id,
                'bbox': (cx, cy, w, h),
                'keypoints': keypoints
            })
    
    return detections


def determine_orientation(keypoints):
    """
    Determine cow orientation based on keypoint positions.
    
    Returns:
        'left': Cow facing left (head on left side of image)
        'right': Cow facing right (head on right side of image)
        'unknown': Cannot determine
    """
    if len(keypoints) < 11:
        return 'unknown'
    
    # Get head keypoints (0: nose, 1: head_neck)
    head_kps = []
    for idx in [0, 1]:
        if idx < len(keypoints):
            x, y, v = keypoints[idx]
            if v > 0:  # visible
                head_kps.append(x)
    
    # Get rear keypoints (4: hip_left, 10: hip_right, 3: mid_back)
    rear_kps = []
    for idx in [4, 10, 3]:
        if idx < len(keypoints):
            x, y, v = keypoints[idx]
            if v > 0:  # visible
                rear_kps.append(x)
    
    if not head_kps or not rear_kps:
        return 'unknown'
    
    avg_head_x = sum(head_kps) / len(head_kps)
    avg_rear_x = sum(rear_kps) / len(rear_kps)
    
    # Threshold for determining orientation (5% of image width)
    threshold = 0.05
    
    if avg_head_x < avg_rear_x - threshold:
        return 'left'  # Head is to the left of rear
    elif avg_head_x > avg_rear_x + threshold:
        return 'right'  # Head is to the right of rear
    else:
        return 'unknown'


def analyze_dataset(labels_dir, dataset_name):
    """Analyze all label files in a directory."""
    if not labels_dir.exists():
        print(f"  {dataset_name}: Directory not found")
        return {}
    
    orientation_counts = defaultdict(int)
    files_analyzed = 0
    
    for label_file in labels_dir.glob("*.txt"):
        detections = parse_yolo_pose_label(label_file)
        files_analyzed += 1
        
        for det in detections:
            orientation = determine_orientation(det['keypoints'])
            orientation_counts[orientation] += 1
    
    print(f"\n  {dataset_name}: {files_analyzed} label files")
    print(f"    Facing LEFT:    {orientation_counts['left']:4d} ({orientation_counts['left'] / max(1, sum(orientation_counts.values())) * 100:.1f}%)")
    print(f"    Facing RIGHT:   {orientation_counts['right']:4d} ({orientation_counts['right'] / max(1, sum(orientation_counts.values())) * 100:.1f}%)")
    print(f"    Unknown:        {orientation_counts['unknown']:4d} ({orientation_counts['unknown'] / max(1, sum(orientation_counts.values())) * 100:.1f}%)")
    
    return orientation_counts


def main():
    print("=" * 60)
    print("COW ORIENTATION ANALYSIS - Training Data Bias Check")
    print("=" * 60)
    
    total_counts = defaultdict(int)
    
    for labels_dir, name in [
        (TRAIN_LABELS, "Training set"),
        (VALID_LABELS, "Validation set"),
        (TEST_LABELS, "Test set"),
    ]:
        counts = analyze_dataset(labels_dir, name)
        for k, v in counts.items():
            total_counts[k] += v
    
    total = sum(total_counts.values())
    print("\n" + "=" * 60)
    print("TOTAL ACROSS ALL DATASETS:")
    print("=" * 60)
    print(f"  Facing LEFT:    {total_counts['left']:4d} ({total_counts['left'] / max(1, total) * 100:.1f}%)")
    print(f"  Facing RIGHT:   {total_counts['right']:4d} ({total_counts['right'] / max(1, total) * 100:.1f}%)")
    print(f"  Unknown:        {total_counts['unknown']:4d} ({total_counts['unknown'] / max(1, total) * 100:.1f}%)")
    print(f"  TOTAL COWS:     {total}")
    
    # Check for bias
    if total > 0:
        left_pct = total_counts['left'] / total * 100
        right_pct = total_counts['right'] / total * 100
        
        print("\n" + "=" * 60)
        print("BIAS ASSESSMENT:")
        print("=" * 60)
        
        if abs(left_pct - right_pct) < 10:
            print("  ✅ Dataset is BALANCED - both directions well represented")
            print("  ℹ️  If pose is wrong, issue is likely skeleton mapping, not orientation bias")
        elif left_pct > right_pct + 10:
            print(f"  ⚠️  Dataset is BIASED toward LEFT-facing cows ({left_pct:.1f}% vs {right_pct:.1f}%)")
            print("  ℹ️  Model may perform worse on right-facing cows")
        else:
            print(f"  ⚠️  Dataset is BIASED toward RIGHT-facing cows ({right_pct:.1f}% vs {left_pct:.1f}%)")
            print("  ℹ️  Model may perform worse on left-facing cows")


if __name__ == "__main__":
    main()

