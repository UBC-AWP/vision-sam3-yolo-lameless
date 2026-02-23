#!/usr/bin/env python3
"""
Cow Pose Data Collection and Annotation Helper

This script helps collect and prepare training data for cow pose estimation
specifically optimized for lameness detection.

Keypoints for Lameness Detection (8 points):
0: head        - Center of poll/head
1: withers     - Top of shoulders (highest point of back)
2: back        - Center of back/spine
3: hip         - Hip bone (tuber coxae)  
4: tailhead    - Base of tail
5: front_hoof  - Front leg hoof (closest to camera)
6: rear_hoof   - Rear leg hoof (closest to camera)
7: belly       - Bottom of belly (for height reference)

For side-view videos, we track the visible legs.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Keypoint definitions for lameness detection
KEYPOINT_NAMES = [
    "head",
    "withers", 
    "back",
    "hip",
    "tailhead",
    "front_hoof",
    "rear_hoof",
    "belly"
]

KEYPOINT_COLORS = {
    "head": (0, 255, 255),      # Yellow
    "withers": (0, 255, 0),     # Green
    "back": (0, 255, 0),        # Green
    "hip": (0, 255, 0),         # Green
    "tailhead": (0, 165, 255),  # Orange
    "front_hoof": (255, 0, 0),  # Blue
    "rear_hoof": (255, 0, 255), # Magenta
    "belly": (0, 255, 255),     # Yellow
}

# Skeleton connections
SKELETON = [
    ("head", "withers"),
    ("withers", "back"),
    ("back", "hip"),
    ("hip", "tailhead"),
    ("withers", "front_hoof"),
    ("hip", "rear_hoof"),
    ("withers", "belly"),
    ("hip", "belly"),
]


class CowAnnotator:
    """Interactive tool for annotating cow keypoints in video frames."""
    
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_frame = 0
        self.annotations = {}
        self.current_keypoint_idx = 0
        self.temp_keypoints = {}
        
        # Load existing annotations if any
        self.annotations_file = self.output_dir / f"{Path(video_path).stem}_annotations.json"
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                self.annotations = json.load(f)
            print(f"Loaded {len(self.annotations)} existing annotations")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for keypoint annotation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            kp_name = KEYPOINT_NAMES[self.current_keypoint_idx]
            self.temp_keypoints[kp_name] = {"x": x, "y": y, "visible": 1}
            print(f"  Marked {kp_name} at ({x}, {y})")
            
            # Move to next keypoint
            self.current_keypoint_idx = (self.current_keypoint_idx + 1) % len(KEYPOINT_NAMES)
    
    def draw_annotations(self, frame, keypoints):
        """Draw keypoints and skeleton on frame."""
        # Draw skeleton connections
        for start, end in SKELETON:
            if start in keypoints and end in keypoints:
                pt1 = (int(keypoints[start]["x"]), int(keypoints[start]["y"]))
                pt2 = (int(keypoints[end]["x"]), int(keypoints[end]["y"]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for name, kp in keypoints.items():
            if kp.get("visible", 1) > 0:
                x, y = int(kp["x"]), int(kp["y"])
                color = KEYPOINT_COLORS.get(name, (255, 255, 255))
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
                cv2.putText(frame, name, (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run(self):
        """Run the annotation tool."""
        cv2.namedWindow("Cow Pose Annotator")
        cv2.setMouseCallback("Cow Pose Annotator", self.mouse_callback)
        
        print("\n" + "="*60)
        print("COW POSE ANNOTATION TOOL")
        print("="*60)
        print("\nKeypoints to annotate:")
        for i, name in enumerate(KEYPOINT_NAMES):
            print(f"  {i}: {name}")
        print("\nControls:")
        print("  LEFT CLICK  - Mark current keypoint")
        print("  SPACE/ENTER - Save frame annotations and go to next")
        print("  N           - Skip to next frame (every 5th)")
        print("  P           - Go to previous frame")
        print("  S           - Save all annotations")
        print("  R           - Reset current frame annotations")
        print("  Q/ESC       - Quit")
        print("="*60 + "\n")
        
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            
            if not ret:
                print("End of video")
                break
            
            display = frame.copy()
            
            # Draw existing annotations for this frame
            frame_key = str(self.current_frame)
            if frame_key in self.annotations:
                display = self.draw_annotations(display, self.annotations[frame_key])
            
            # Draw temporary keypoints being annotated
            if self.temp_keypoints:
                display = self.draw_annotations(display, self.temp_keypoints)
            
            # Draw UI
            cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Current: {KEYPOINT_NAMES[self.current_keypoint_idx]}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Annotated frames: {len(self.annotations)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Cow Pose Annotator", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' ') or key == 13:  # Space or Enter
                if self.temp_keypoints:
                    self.annotations[frame_key] = self.temp_keypoints.copy()
                    print(f"Saved annotations for frame {self.current_frame}")
                self.temp_keypoints = {}
                self.current_keypoint_idx = 0
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            elif key == ord('n'):  # Next (skip 5 frames)
                self.temp_keypoints = {}
                self.current_keypoint_idx = 0
                self.current_frame = min(self.current_frame + 5, self.total_frames - 1)
            elif key == ord('p'):  # Previous
                self.temp_keypoints = {}
                self.current_keypoint_idx = 0
                self.current_frame = max(self.current_frame - 1, 0)
            elif key == ord('s'):  # Save
                self.save_annotations()
            elif key == ord('r'):  # Reset
                self.temp_keypoints = {}
                self.current_keypoint_idx = 0
                if frame_key in self.annotations:
                    del self.annotations[frame_key]
                print(f"Reset annotations for frame {self.current_frame}")
        
        self.save_annotations()
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_annotations(self):
        """Save annotations to JSON file."""
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"\nSaved {len(self.annotations)} annotated frames to {self.annotations_file}")
    
    def export_to_yolo(self):
        """Export annotations to YOLO pose format."""
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        for frame_idx, keypoints in self.annotations.items():
            frame_num = int(frame_idx)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
            
            # Save image
            img_name = f"{Path(self.video_path).stem}_{frame_num:06d}.jpg"
            cv2.imwrite(str(images_dir / img_name), frame)
            
            # Calculate bounding box from keypoints
            xs = [kp["x"] for kp in keypoints.values() if kp.get("visible", 1) > 0]
            ys = [kp["y"] for kp in keypoints.values() if kp.get("visible", 1) > 0]
            
            if not xs or not ys:
                continue
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Add margin
            margin = 0.1
            w = x_max - x_min
            h = y_max - y_min
            x_min = max(0, x_min - w * margin)
            x_max = min(self.width, x_max + w * margin)
            y_min = max(0, y_min - h * margin)
            y_max = min(self.height, y_max + h * margin)
            
            # Convert to YOLO format (normalized)
            cx = (x_min + x_max) / 2 / self.width
            cy = (y_min + y_max) / 2 / self.height
            bw = (x_max - x_min) / self.width
            bh = (y_max - y_min) / self.height
            
            # Build YOLO pose line: class cx cy w h kp1_x kp1_y kp1_v ...
            line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            
            for kp_name in KEYPOINT_NAMES:
                if kp_name in keypoints:
                    kp = keypoints[kp_name]
                    kx = kp["x"] / self.width
                    ky = kp["y"] / self.height
                    kv = kp.get("visible", 1)
                    line += f" {kx:.6f} {ky:.6f} {kv}"
                else:
                    line += " 0 0 0"
            
            # Save label
            label_name = f"{Path(self.video_path).stem}_{frame_num:06d}.txt"
            with open(labels_dir / label_name, 'w') as f:
                f.write(line + "\n")
        
        # Create dataset YAML
        yaml_content = f"""# Cow Pose Dataset for Lameness Detection
# Generated: {datetime.now().isoformat()}

path: {self.output_dir.absolute()}
train: images
val: images

# Keypoints
kpt_shape: [{len(KEYPOINT_NAMES)}, 3]

# Classes
names:
  0: cow

# Keypoint names
keypoint_names: {KEYPOINT_NAMES}

# Skeleton connections (for visualization)
skeleton:
{chr(10).join(f'  - [{KEYPOINT_NAMES.index(s)}, {KEYPOINT_NAMES.index(e)}]' for s, e in SKELETON)}
"""
        
        with open(self.output_dir / "cow_pose.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"\nExported {len(self.annotations)} frames to YOLO format")
        print(f"  Images: {images_dir}")
        print(f"  Labels: {labels_dir}")
        print(f"  Config: {self.output_dir / 'cow_pose.yaml'}")


def extract_frames_for_annotation(video_path: str, output_dir: str, interval: int = 5):
    """Extract frames from video at regular intervals for annotation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    extracted = 0
    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            img_path = output_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(img_path), frame)
            extracted += 1
    
    cap.release()
    print(f"Extracted {extracted} frames to {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Cow Pose Data Collection Tool")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="data/cow_pose_custom",
                       help="Output directory for annotations")
    parser.add_argument("--extract", "-e", action="store_true",
                       help="Extract frames instead of interactive annotation")
    parser.add_argument("--interval", "-i", type=int, default=5,
                       help="Frame extraction interval")
    parser.add_argument("--export", action="store_true",
                       help="Export existing annotations to YOLO format")
    
    args = parser.parse_args()
    
    if args.extract:
        extract_frames_for_annotation(args.video, args.output, args.interval)
    else:
        annotator = CowAnnotator(args.video, args.output)
        if args.export:
            annotator.export_to_yolo()
        else:
            annotator.run()
            
            # Ask to export
            response = input("\nExport to YOLO format? (y/n): ")
            if response.lower() == 'y':
                annotator.export_to_yolo()


if __name__ == "__main__":
    main()

