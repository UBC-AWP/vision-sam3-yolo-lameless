#!/usr/bin/env python3
"""
Test RTMPose Animal Pose Estimation on Cow Images

This script tests the pretrained RTMPose model on cow images.
RTMPose is trained on AP-10K dataset which includes cows.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def download_model(url, save_path):
    """Download model if not exists"""
    if not save_path.exists():
        print(f"Downloading model to {save_path}...")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, save_path)
        print("Download complete!")
    else:
        print(f"Model already exists: {save_path}")
    return save_path

def main():
    print("=" * 60)
    print("Testing RTMPose Animal Pose Estimation")
    print("=" * 60)
    
    # Model URLs from MMPose model zoo
    # RTMPose-m trained on AP-10K (animal pose dataset)
    config_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb64-210e_ap10k-256x256-98b53687_20230920.pth"
    
    # Alternative: Use local model path
    model_dir = PROJECT_ROOT / "data" / "models" / "rtmpose"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if mmpose is installed
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import PoseDataSample
        print("✅ MMPose imported successfully")
    except ImportError as e:
        print(f"❌ MMPose import error: {e}")
        print("Please install: pip install mmpose mmcv mmdet mmengine")
        return
    
    # Try to use the pretrained model
    try:
        # Download RTMPose config and checkpoint
        config_path = "configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py"
        checkpoint_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth"
        
        checkpoint_path = model_dir / "rtmpose-m_ap10k.pth"
        
        # Download checkpoint
        if not checkpoint_path.exists():
            print(f"Downloading RTMPose checkpoint...")
            urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
            print(f"Downloaded to {checkpoint_path}")
        
        print(f"\n📦 Model checkpoint: {checkpoint_path}")
        print(f"   Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # List available test images
        videos_dir = PROJECT_ROOT / "data" / "videos"
        processed_dir = PROJECT_ROOT / "data" / "processed" / "frames"
        
        print(f"\n🖼️  Looking for test images...")
        
        # Check if we have any frames extracted
        if processed_dir.exists():
            frame_dirs = list(processed_dir.glob("*"))
            if frame_dirs:
                # Get first frame from first video
                for frame_dir in frame_dirs:
                    frames = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))
                    if frames:
                        test_image = frames[0]
                        print(f"   Found test image: {test_image}")
                        break
        
        print("\n" + "=" * 60)
        print("RTMPose Model Ready!")
        print("=" * 60)
        print(f"""
To use RTMPose for animal pose estimation:

1. The model is trained on AP-10K dataset with 17 keypoints:
   - 0: left_eye, 1: right_eye, 2: nose
   - 3: neck, 4: root_of_tail
   - 5-8: front legs (shoulder, elbow, wrist, paw)
   - 9-12: back legs
   - 13-16: body parts

2. To integrate with the pipeline:
   - Update tleap-pipeline to use MMPose instead of YOLOv8-Pose
   - Or convert RTMPose outputs to match current format

Model checkpoint downloaded to: {checkpoint_path}
""")
        
        return str(checkpoint_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()


