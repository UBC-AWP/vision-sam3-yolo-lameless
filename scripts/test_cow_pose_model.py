"""
Test the trained cow pose estimation model
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Paths
MODEL_PATH = Path("/Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/runs/cow_pose/train_v1/weights/best.pt")
TEST_IMAGES_DIR = Path("/Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/data/pose_datasets/cow-pose-yolo/images/val")
OUTPUT_DIR = Path("/Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/runs/cow_pose/test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keypoint names (20 keypoints)
KEYPOINT_NAMES = [
    'left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear',
    'left_front_elbow', 'right_front_elbow', 'left_back_elbow', 'right_back_elbow',
    'left_front_knee', 'right_front_knee', 'left_back_knee', 'right_back_knee',
    'left_front_paw', 'right_front_paw', 'left_back_paw', 'right_back_paw',
    'throat', 'withers', 'tailbase'
]

# Skeleton connections (pairs of keypoint indices)
SKELETON = [
    (0, 1),   # left_eye - right_eye
    (0, 2),   # left_eye - nose
    (1, 2),   # right_eye - nose
    (0, 3),   # left_eye - left_ear
    (1, 4),   # right_eye - right_ear
    (2, 17),  # nose - throat
    (17, 18), # throat - withers
    (18, 19), # withers - tailbase
    (5, 9),   # left_front_elbow - left_front_knee
    (6, 10),  # right_front_elbow - right_front_knee
    (7, 11),  # left_back_elbow - left_back_knee
    (8, 12),  # right_back_elbow - right_back_knee
    (9, 13),  # left_front_knee - left_front_paw
    (10, 14), # right_front_knee - right_front_paw
    (11, 15), # left_back_knee - left_back_paw
    (12, 16), # right_back_knee - right_back_paw
]

# Colors for different body parts
COLORS = {
    "face": (0, 255, 255),       # Yellow - face area
    "spine": (0, 255, 0),        # Green - spine
    "front_left": (255, 0, 0),   # Blue - front left leg
    "front_right": (0, 165, 255),# Orange - front right leg
    "back_left": (255, 0, 255),  # Magenta - back left leg
    "back_right": (0, 255, 255), # Cyan - back right leg
}

def get_limb_color(i):
    """Get color for each skeleton connection"""
    if i < 5:  # Face connections
        return COLORS["face"]
    elif i < 8:  # Spine connections
        return COLORS["spine"]
    elif i in [8, 12]:  # Front left
        return COLORS["front_left"]
    elif i in [9, 13]:  # Front right
        return COLORS["front_right"]
    elif i in [10, 14]:  # Back left
        return COLORS["back_left"]
    else:  # Back right
        return COLORS["back_right"]


def draw_pose(image, keypoints, bbox, conf_threshold=0.3):
    """Draw pose skeleton on image"""
    h, w = image.shape[:2]
    
    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Cow {bbox[4]:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if keypoints is None or len(keypoints) == 0:
        return image
    
    # Draw skeleton lines
    for i, (start_idx, end_idx) in enumerate(SKELETON):
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            kp1 = keypoints[start_idx]
            kp2 = keypoints[end_idx]
            
            # Check confidence
            if len(kp1) >= 3 and len(kp2) >= 3:
                if kp1[2] > conf_threshold and kp2[2] > conf_threshold:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    color = get_limb_color(i)
                    cv2.line(image, pt1, pt2, color, 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if len(kp) >= 3 and kp[2] > conf_threshold:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    
    return image


def main():
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Get test images
    test_images = list(TEST_IMAGES_DIR.glob("*.jpeg")) + list(TEST_IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(test_images)} test images")
    
    # Process first 5 images
    for img_path in test_images[:5]:
        print(f"\nProcessing: {img_path.name}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Failed to read image")
            continue
        
        # Run inference
        results = model(image, verbose=False)
        
        # Draw results
        for result in results:
            # Get boxes
            if result.boxes is not None and len(result.boxes) > 0:
                for j, box in enumerate(result.boxes):
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    bbox_with_conf = np.append(bbox, conf)
                    
                    # Get corresponding keypoints
                    keypoints = None
                    if result.keypoints is not None and j < len(result.keypoints):
                        keypoints = result.keypoints[j].data[0].cpu().numpy()
                    
                    image = draw_pose(image, keypoints, bbox_with_conf)
                    
                    print(f"  Detected cow with confidence {conf:.2f}")
                    if keypoints is not None:
                        visible_kps = sum(1 for kp in keypoints if len(kp) >= 3 and kp[2] > 0.3)
                        print(f"  Visible keypoints: {visible_kps}/{len(KEYPOINT_NAMES)}")
        
        # Save result
        output_path = OUTPUT_DIR / f"result_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"  Saved to: {output_path}")
    
    print(f"\n✅ Test results saved to: {OUTPUT_DIR}")
    print("\nTo view results, check the images in the output directory.")


if __name__ == "__main__":
    main()

