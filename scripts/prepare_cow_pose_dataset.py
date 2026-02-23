"""
Prepare Cow Pose Dataset for YOLOv8-Pose Training
Extracts cow-only data from Animal Pose Dataset and converts to YOLO format
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random
from PIL import Image

# Paths
DATASET_ROOT = Path("/Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/data/pose_datasets/animal-pose-part1")
OUTPUT_DIR = Path("/Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/data/pose_datasets/cow-pose-yolo")

# Cow category ID
COW_CATEGORY_ID = 5

# Keypoint names (20 keypoints)
KEYPOINT_NAMES = [
    'left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear',
    'left_front_elbow', 'right_front_elbow', 'left_back_elbow', 'right_back_elbow',
    'left_front_knee', 'right_front_knee', 'left_back_knee', 'right_back_knee',
    'left_front_paw', 'right_front_paw', 'left_back_paw', 'right_back_paw',
    'throat', 'withers', 'tailbase'
]


def convert_to_yolo_pose(annotation, img_width, img_height):
    """Convert annotation to YOLO pose format"""
    # Get bounding box (format: [x, y, width, height])
    bbox = annotation['bbox']
    x_center = (bbox[0] + bbox[2] / 2) / img_width
    y_center = (bbox[1] + bbox[3] / 2) / img_height
    width = bbox[2] / img_width
    height = bbox[3] / img_height
    
    # Get keypoints (format: [[x1, y1, v1], [x2, y2, v2], ...])
    keypoints = annotation['keypoints']
    
    # Convert keypoints to YOLO format: x y visibility (normalized)
    yolo_keypoints = []
    for kp in keypoints:
        x = kp[0] / img_width if kp[0] > 0 else 0
        y = kp[1] / img_height if kp[1] > 0 else 0
        v = kp[2]  # visibility: 0=not labeled, 1=labeled
        yolo_keypoints.extend([x, y, v])
    
    # YOLO pose format: class x_center y_center width height kp1_x kp1_y kp1_v ...
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(f"{v:.6f}" for v in yolo_keypoints)


def main():
    print("Loading Animal Pose Dataset...")
    
    # Load annotations
    with open(DATASET_ROOT / "keypoints.json") as f:
        data = json.load(f)
    
    # Images dict: {id: filename}
    images = data['images']  # {id_str: filename}
    
    # Filter cow annotations
    cow_annotations = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] == COW_CATEGORY_ID:
            img_id = str(ann['image_id'])  # Convert to string to match dict keys
            cow_annotations[img_id].append(ann)
    
    print(f"Found {len(cow_annotations)} images with cow annotations")
    print(f"Total cow instances: {sum(len(v) for v in cow_annotations.values())}")
    
    # Create output directories
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Split data (80% train, 20% val)
    image_ids = list(cow_annotations.keys())
    random.seed(42)
    random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * 0.8)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])
    
    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    
    # Process images
    processed = {'train': 0, 'val': 0}
    skipped = 0
    
    for img_id, annotations in cow_annotations.items():
        img_filename = images.get(img_id)
        if not img_filename:
            skipped += 1
            continue
        
        # Determine split
        split = 'train' if img_id in train_ids else 'val'
        
        # Find source image in cow subfolder
        src_path = DATASET_ROOT / "animalpose_image_part2" / "cow" / img_filename
        
        if not src_path.exists():
            skipped += 1
            continue
        
        # Get image dimensions
        try:
            with Image.open(src_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading {src_path}: {e}")
            skipped += 1
            continue
        
        # Copy image
        dst_img_path = OUTPUT_DIR / 'images' / split / img_filename
        shutil.copy(src_path, dst_img_path)
        
        # Create label file
        label_filename = Path(img_filename).stem + '.txt'
        dst_label_path = OUTPUT_DIR / 'labels' / split / label_filename
        
        with open(dst_label_path, 'w') as f:
            for ann in annotations:
                yolo_line = convert_to_yolo_pose(ann, img_width, img_height)
                f.write(yolo_line + '\n')
        
        processed[split] += 1
    
    print(f"\nProcessed: Train={processed['train']}, Val={processed['val']}")
    print(f"Skipped (image not found): {skipped}")
    
    # Create dataset YAML for YOLOv8
    yaml_content = f"""# Cow Pose Estimation Dataset
# Animal Pose Dataset - Cow subset

path: {OUTPUT_DIR}
train: images/train
val: images/val

# Keypoints
kpt_shape: [20, 3]  # 20 keypoints, each with x, y, visibility

# Classes
names:
  0: cow

# Flip indices for horizontal flip augmentation
# Maps left keypoints to right keypoints
flip_idx: [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19]
"""
    
    yaml_path = OUTPUT_DIR / 'cow_pose.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset YAML saved to: {yaml_path}")
    print("\nKeypoints (20 total):")
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {i}: {name}")
    
    print("\n" + "="*60)
    print("To train YOLOv8-Pose model, run:")
    print(f"  yolo pose train data={yaml_path} model=yolov8n-pose.pt epochs=100 imgsz=640")
    print("="*60)


if __name__ == "__main__":
    main()
