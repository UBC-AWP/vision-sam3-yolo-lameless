#!/usr/bin/env python3
"""
Prepare combined cow pose dataset from AP-10K and Kaggle datasets.
Converts AP-10K COCO format to YOLO pose format and combines with Kaggle dataset.
"""

import json
import shutil
from pathlib import Path
import cv2
import yaml
from sklearn.model_selection import train_test_split
import random

# Paths
AP10K_DIR = Path("data/ap-10k")
KAGGLE_DIR = Path("data/Cow Pose Estimation")
OUTPUT_DIR = Path("data/cow_pose_combined")

# AP-10K keypoints (17)
AP10K_KEYPOINTS = [
    'left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail',
    'left_shoulder', 'left_elbow', 'left_front_paw',
    'right_shoulder', 'right_elbow', 'right_front_paw',
    'left_hip', 'left_knee', 'left_back_paw',
    'right_hip', 'right_knee', 'right_back_paw'
]

# We'll use AP-10K format (17 keypoints) as the standard
# Kaggle dataset has 12 keypoints - we'll pad with zeros for missing ones


def convert_ap10k_to_yolo():
    """Convert AP-10K cow data to YOLO pose format."""
    print("Converting AP-10K cow data to YOLO format...")
    
    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    
    # Process all splits
    for split_file in AP10K_DIR.glob("annotations/*.json"):
        with open(split_file) as f:
            data = json.load(f)
        
        # Build image lookup
        images = {img['id']: img for img in data['images']}
        
        # Get cow annotations (category_id = 5)
        cow_annotations = [a for a in data['annotations'] if a['category_id'] == 5]
        
        # Group by image
        img_to_anns = {}
        for ann in cow_annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        for img_id, anns in img_to_anns.items():
            img_info = images[img_id]
            img_file = img_info['file_name']
            img_w, img_h = img_info['width'], img_info['height']
            
            # Source image path
            src_img = AP10K_DIR / "data" / img_file
            if not src_img.exists():
                continue
            
            # Copy image
            dst_img = output_images / f"ap10k_{img_file}"
            if not dst_img.exists():
                shutil.copy(src_img, dst_img)
            
            # Convert annotations to YOLO format
            yolo_lines = []
            for ann in anns:
                bbox = ann['bbox']  # [x, y, w, h] in pixels
                keypoints = ann['keypoints']  # [x, y, v, x, y, v, ...]
                
                # Convert bbox to YOLO format (normalized cx, cy, w, h)
                cx = (bbox[0] + bbox[2] / 2) / img_w
                cy = (bbox[1] + bbox[3] / 2) / img_h
                bw = bbox[2] / img_w
                bh = bbox[3] / img_h
                
                # Convert keypoints to YOLO format
                kp_str = ""
                for i in range(0, len(keypoints), 3):
                    kx = keypoints[i] / img_w
                    ky = keypoints[i + 1] / img_h
                    kv = keypoints[i + 2]  # 0=not labeled, 1=labeled but not visible, 2=visible
                    kp_str += f" {kx:.6f} {ky:.6f} {kv}"
                
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}{kp_str}")
            
            # Write label file
            label_file = output_labels / f"ap10k_{Path(img_file).stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            converted += 1
    
    print(f"  Converted {converted} AP-10K cow images")
    return converted


def copy_kaggle_dataset():
    """Copy Kaggle cow pose dataset to combined directory."""
    print("Copying Kaggle cow pose dataset...")
    
    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"
    
    copied = 0
    
    for split in ['train', 'val']:
        img_dir = KAGGLE_DIR / "images" / split
        label_dir = KAGGLE_DIR / "labels" / split
        
        if not img_dir.exists():
            continue
        
        for img_file in img_dir.glob("*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Read original label to check keypoint count
            label_file = label_dir / f"{img_file.stem}.txt"
            
            # Copy image
            dst_img = output_images / f"kaggle_{img_file.name}"
            shutil.copy(img_file, dst_img)
            
            # Handle label - need to pad to 17 keypoints if it has 12
            if label_file.exists():
                with open(label_file) as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    # Check current keypoint count
                    bbox_parts = 5
                    kp_parts = len(parts) - bbox_parts
                    current_kp = kp_parts // 3
                    
                    if current_kp == 12:
                        # Pad to 17 keypoints (add 5 more with zeros)
                        # Kaggle 12 kp might be subset of AP-10K 17 kp
                        # For now, add zeros for missing keypoints
                        padding = " 0 0 0" * (17 - 12)
                        new_lines.append(line.strip() + padding)
                    else:
                        new_lines.append(line.strip())
                
                dst_label = output_labels / f"kaggle_{img_file.stem}.txt"
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(new_lines))
            
            copied += 1
    
    print(f"  Copied {copied} Kaggle cow images")
    return copied


def split_dataset():
    """Split combined dataset into train/val."""
    print("Splitting dataset into train/val...")
    
    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels"
    
    # Get all images
    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    
    # Split 80/20
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # Create split directories
    for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
        split_img_dir = OUTPUT_DIR / "images" / split
        split_label_dir = OUTPUT_DIR / "labels" / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img in imgs:
            # Move image
            shutil.move(str(img), split_img_dir / img.name)
            
            # Move label
            label_file = labels_dir / f"{img.stem}.txt"
            if label_file.exists():
                shutil.move(str(label_file), split_label_dir / label_file.name)
    
    # Clean up root images/labels dirs
    # (they should be empty now, but just in case)
    for f in images_dir.glob("*"):
        if f.is_file():
            f.unlink()
    for f in labels_dir.glob("*"):
        if f.is_file():
            f.unlink()
    
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val: {len(val_imgs)} images")
    
    return len(train_imgs), len(val_imgs)


def create_dataset_yaml():
    """Create YOLO dataset configuration file."""
    print("Creating dataset YAML...")
    
    # Skeleton connections (1-indexed for YOLO)
    skeleton = [
        [1, 2],   # left_eye - right_eye
        [1, 3],   # left_eye - nose
        [2, 3],   # right_eye - nose
        [3, 4],   # nose - neck
        [4, 5],   # neck - root_of_tail
        [4, 6],   # neck - left_shoulder
        [6, 7],   # left_shoulder - left_elbow
        [7, 8],   # left_elbow - left_front_paw
        [4, 9],   # neck - right_shoulder
        [9, 10],  # right_shoulder - right_elbow
        [10, 11], # right_elbow - right_front_paw
        [5, 12],  # root_of_tail - left_hip
        [12, 13], # left_hip - left_knee
        [13, 14], # left_knee - left_back_paw
        [5, 15],  # root_of_tail - right_hip
        [15, 16], # right_hip - right_knee
        [16, 17], # right_knee - right_back_paw
    ]
    
    config = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'cow'},
        'kpt_shape': [17, 3],  # 17 keypoints, 3 values each (x, y, visibility)
        'flip_idx': [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13],  # For horizontal flip augmentation
        'skeleton': skeleton,
    }
    
    yaml_path = OUTPUT_DIR / "cow_pose.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Also create a keypoints reference file
    kp_ref = OUTPUT_DIR / "keypoints_reference.txt"
    with open(kp_ref, 'w') as f:
        f.write("Keypoint Index Reference (0-indexed):\n")
        f.write("=" * 40 + "\n")
        for i, kp in enumerate(AP10K_KEYPOINTS):
            f.write(f"  {i:2d}: {kp}\n")
        f.write("\n")
        f.write("Skeleton Connections:\n")
        f.write("=" * 40 + "\n")
        for conn in skeleton:
            f.write(f"  {AP10K_KEYPOINTS[conn[0]-1]} -- {AP10K_KEYPOINTS[conn[1]-1]}\n")
    
    print(f"  Created {yaml_path}")
    print(f"  Created {kp_ref}")
    
    return yaml_path


def main():
    print("=" * 60)
    print("PREPARING COMBINED COW POSE DATASET")
    print("=" * 60)
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Convert and copy datasets
    ap10k_count = convert_ap10k_to_yolo()
    kaggle_count = copy_kaggle_dataset()
    
    # Split into train/val
    train_count, val_count = split_dataset()
    
    # Create YAML config
    yaml_path = create_dataset_yaml()
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Total images: {ap10k_count + kaggle_count}")
    print(f"  - AP-10K: {ap10k_count}")
    print(f"  - Kaggle: {kaggle_count}")
    print(f"Train/Val split: {train_count}/{val_count}")
    print(f"\nDataset config: {yaml_path}")
    print("\nTo train YOLOv8-Pose:")
    print(f"  yolo pose train data={yaml_path} model=yolov8n-pose.pt epochs=100 device=mps")
    print("=" * 60)


if __name__ == "__main__":
    main()


