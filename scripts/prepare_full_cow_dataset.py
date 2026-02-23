#!/usr/bin/env python3
"""
Prepare Full Cow Pose Dataset

Combines ALL available cow pose data:
1. AP-10K dataset (all 3 splits) - 200 unique cow images with 17 keypoints
2. Kaggle Cow Pose Estimation - 181 images with 12 keypoints

Both datasets use the same keypoint format after conversion.
"""

import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "cow_pose_full"

# AP-10K keypoint names (17 keypoints)
AP10K_KEYPOINTS = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw", 
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw"
]

# Kaggle dataset has 12 keypoints, map to AP-10K format (17 keypoints)
# Kaggle keypoints: nose, L_eye, R_eye, L_ear, R_ear, L_shoulder, R_shoulder,
#                  L_elbow, R_elbow, L_hip, R_hip, tail
KAGGLE_TO_AP10K = {
    0: 2,   # nose -> Nose
    1: 0,   # L_eye -> L_Eye
    2: 1,   # R_eye -> R_Eye
    3: None,  # L_ear (no mapping)
    4: None,  # R_ear (no mapping)
    5: 5,   # L_shoulder -> L_Shoulder
    6: 8,   # R_shoulder -> R_Shoulder
    7: 6,   # L_elbow -> L_Elbow
    8: 9,   # R_elbow -> R_Elbow
    9: 11,  # L_hip -> L_Hip
    10: 14, # R_hip -> R_Hip
    11: 4   # tail -> Root_of_tail
}


def process_ap10k_cow_data():
    """Extract cow data from AP-10K dataset"""
    print("\n" + "=" * 60)
    print("Processing AP-10K Cow Data")
    print("=" * 60)
    
    ap10k_dir = PROJECT_ROOT / "data" / "ap-10k"
    annotations_dir = ap10k_dir / "annotations"
    images_dir = ap10k_dir / "data"
    
    all_images = {}
    all_annotations = []
    
    # Process all splits
    for split_type in ['train', 'val', 'test']:
        for split_num in [1, 2, 3]:
            json_file = annotations_dir / f"ap10k-{split_type}-split{split_num}.json"
            if not json_file.exists():
                continue
                
            with open(json_file) as f:
                data = json.load(f)
            
            # Get cow category ID
            cow_cat_id = None
            for cat in data['categories']:
                if cat['name'].lower() == 'cow':
                    cow_cat_id = cat['id']
                    break
            
            if cow_cat_id is None:
                continue
            
            # Map image IDs to filenames
            img_id_to_info = {img['id']: img for img in data['images']}
            
            # Get cow annotations
            for ann in data['annotations']:
                if ann['category_id'] == cow_cat_id:
                    img_id = ann['image_id']
                    if img_id in img_id_to_info:
                        img_info = img_id_to_info[img_id]
                        all_images[img_id] = img_info
                        all_annotations.append(ann)
    
    print(f"Found {len(all_images)} unique cow images")
    print(f"Found {len(all_annotations)} cow annotations")
    
    return all_images, all_annotations, images_dir


def process_kaggle_cow_data():
    """Process Kaggle Cow Pose Estimation dataset"""
    print("\n" + "=" * 60)
    print("Processing Kaggle Cow Pose Data")
    print("=" * 60)
    
    kaggle_dir = PROJECT_ROOT / "data" / "Cow Pose Estimation"
    
    all_images = []
    all_labels = []
    
    for split in ['train', 'val']:
        img_dir = kaggle_dir / "images" / split
        lbl_dir = kaggle_dir / "labels" / split
        
        if not img_dir.exists():
            continue
            
        for img_file in img_dir.glob("*.jpg"):
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                all_images.append((split, img_file))
                with open(lbl_file) as f:
                    labels = f.read().strip()
                all_labels.append(labels)
    
    print(f"Found {len(all_images)} Kaggle cow images")
    
    return all_images, all_labels


def convert_ap10k_to_yolo(annotation, img_info, img_path):
    """Convert AP-10K annotation to YOLO format"""
    keypoints = annotation['keypoints']
    bbox = annotation['bbox']  # COCO format: [x, y, width, height]
    
    # Get image dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_h, img_w = img.shape[:2]
    
    # Convert bbox to YOLO format (center_x, center_y, width, height) normalized
    x, y, w, h = bbox
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    
    # Convert keypoints to YOLO format
    # Format: class_id cx cy w h kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis ...
    yolo_keypoints = []
    for i in range(0, len(keypoints), 3):
        kp_x, kp_y, kp_vis = keypoints[i:i+3]
        # Normalize
        kp_x_norm = kp_x / img_w
        kp_y_norm = kp_y / img_h
        # Visibility: 0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible
        kp_vis_yolo = 2 if kp_vis > 0 else 0
        yolo_keypoints.extend([kp_x_norm, kp_y_norm, kp_vis_yolo])
    
    # Format: class cx cy w h kp1_x kp1_y kp1_v ...
    label_line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
    for kp in yolo_keypoints:
        label_line += f" {kp:.6f}" if isinstance(kp, float) else f" {kp}"
    
    return label_line, (img_w, img_h)


def convert_kaggle_to_17_keypoints(kaggle_label):
    """Convert 12-keypoint Kaggle format to 17-keypoint format"""
    parts = kaggle_label.strip().split()
    if len(parts) < 5:
        return None
    
    class_id = parts[0]
    cx, cy, w, h = map(float, parts[1:5])
    
    # Parse 12 keypoints
    keypoints_12 = []
    for i in range(5, len(parts), 3):
        if i+2 < len(parts):
            kx, ky, kv = float(parts[i]), float(parts[i+1]), int(float(parts[i+2]))
            keypoints_12.append((kx, ky, kv))
    
    # Convert to 17 keypoints
    keypoints_17 = [(0.0, 0.0, 0) for _ in range(17)]  # Initialize with invisible
    
    for kaggle_idx, ap10k_idx in KAGGLE_TO_AP10K.items():
        if ap10k_idx is not None and kaggle_idx < len(keypoints_12):
            keypoints_17[ap10k_idx] = keypoints_12[kaggle_idx]
    
    # Build label line
    label_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    for kx, ky, kv in keypoints_17:
        label_line += f" {kx:.6f} {ky:.6f} {kv}"
    
    return label_line


def create_dataset():
    """Create the full combined dataset"""
    print("\n" + "=" * 60)
    print("Creating Full Cow Pose Dataset")
    print("=" * 60)
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    (OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    train_count = 0
    val_count = 0
    
    # Process AP-10K data
    ap10k_images, ap10k_annotations, ap10k_img_dir = process_ap10k_cow_data()
    
    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in ap10k_annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # Convert and save AP-10K data (80% train, 20% val)
    img_ids = list(ap10k_images.keys())
    np.random.seed(42)
    np.random.shuffle(img_ids)
    
    split_idx = int(len(img_ids) * 0.8)
    train_ids = set(img_ids[:split_idx])
    
    for img_id, img_info in ap10k_images.items():
        img_file = ap10k_img_dir / img_info['file_name']
        if not img_file.exists():
            continue
        
        split = 'train' if img_id in train_ids else 'val'
        
        # Convert all annotations for this image
        label_lines = []
        for ann in img_to_anns[img_id]:
            result = convert_ap10k_to_yolo(ann, img_info, img_file)
            if result:
                label_line, _ = result
                label_lines.append(label_line)
        
        if label_lines:
            # Copy image
            dst_img = OUTPUT_DIR / split / "images" / f"ap10k_{img_id}.jpg"
            shutil.copy(img_file, dst_img)
            
            # Save labels
            dst_lbl = OUTPUT_DIR / split / "labels" / f"ap10k_{img_id}.txt"
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(label_lines))
            
            if split == 'train':
                train_count += 1
            else:
                val_count += 1
    
    print(f"Processed AP-10K: {train_count} train, {val_count} val")
    
    # Process Kaggle data
    kaggle_images, kaggle_labels = process_kaggle_cow_data()
    
    kaggle_train = 0
    kaggle_val = 0
    
    for (orig_split, img_file), label in zip(kaggle_images, kaggle_labels):
        # Convert label to 17 keypoints
        converted_label = convert_kaggle_to_17_keypoints(label)
        if converted_label is None:
            continue
        
        split = orig_split  # Keep original train/val split
        
        # Copy image with unique name
        dst_img = OUTPUT_DIR / split / "images" / f"kaggle_{img_file.name}"
        shutil.copy(img_file, dst_img)
        
        # Save label
        dst_lbl = OUTPUT_DIR / split / "labels" / f"kaggle_{img_file.stem}.txt"
        with open(dst_lbl, 'w') as f:
            f.write(converted_label)
        
        if split == 'train':
            train_count += 1
            kaggle_train += 1
        else:
            val_count += 1
            kaggle_val += 1
    
    print(f"Processed Kaggle: {kaggle_train} train, {kaggle_val} val")
    
    # Create dataset YAML
    yaml_content = f"""# Full Cow Pose Dataset
# Combined from AP-10K and Kaggle Cow Pose Estimation

path: {OUTPUT_DIR}
train: train/images
val: val/images

# 17 keypoints matching AP-10K format
kpt_shape: [17, 3]

# Flip mapping for data augmentation
flip_idx: [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]

# Classes
names:
  0: cow

# Keypoint names (for reference)
# 0: L_Eye, 1: R_Eye, 2: Nose, 3: Neck, 4: Root_of_tail
# 5: L_Shoulder, 6: L_Elbow, 7: L_F_Paw
# 8: R_Shoulder, 9: R_Elbow, 10: R_F_Paw
# 11: L_Hip, 12: L_Knee, 13: L_B_Paw
# 14: R_Hip, 15: R_Knee, 16: R_B_Paw
"""
    
    yaml_path = OUTPUT_DIR / "cow_pose_full.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n" + "=" * 60)
    print("Dataset Created!")
    print("=" * 60)
    print(f"Total: {train_count} train, {val_count} val")
    print(f"YAML config: {yaml_path}")
    print(f"\nTo train:")
    print(f"yolo pose train data={yaml_path} model=yolov8m-pose.pt epochs=150 device=mps")


if __name__ == "__main__":
    create_dataset()


