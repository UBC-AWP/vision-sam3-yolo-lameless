"""
Video Preprocessing Service
Detects cows using YOLO and crops videos to show only the cow
"""
import asyncio
import os
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import yaml
from shared.utils.nats_client import NATSClient


class VideoPreprocessor:
    """Preprocess videos by detecting and cropping cows"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Initialize YOLO model (use pretrained COCO model initially)
        # In production, use custom trained cow detection model
        self.yolo_model = YOLO("yolov8n.pt")  # Start with nano model for speed
        
        # Directories
        self.videos_dir = Path("/app/data/videos")
        self.processed_dir = Path("/app/data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load configuration"""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    async def process_video(self, video_data: dict):
        """Process a video: detect cow and crop"""
        video_id = video_data["video_id"]
        input_path = Path(video_data["file_path"])
        
        print(f"Processing video {video_id}: {input_path}")
        
        if not input_path.exists():
            print(f"Video file not found: {input_path}")
            return
        
        # Output path for cropped video
        output_path = self.processed_dir / f"{video_id}_cropped.mp4"
        
        try:
            # Load video
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {input_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Detect cow in first few frames to get bounding box
            cow_boxes = []
            sample_frames = min(10, total_frames)
            
            for i in range(sample_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO detection
                results = self.yolo_model(frame, verbose=False)
                
                # Find largest "cow" detection (class 19 in COCO, or custom class)
                # For now, we'll look for any large animal-like detection
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Filter by confidence and size
                        if box.conf[0] > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            w, h = x2 - x1, y2 - y1
                            # Filter large detections (likely animals)
                            if w * h > (width * height * 0.1):  # At least 10% of frame
                                cow_boxes.append([x1, y1, x2, y2])
            
            cap.release()
            
            if not cow_boxes:
                print(f"No cow detected in video {video_id}")
                # Use full frame as fallback
                crop_box = [0, 0, width, height]
            else:
                # Use median bounding box
                cow_boxes = np.array(cow_boxes)
                crop_box = [
                    int(np.median(cow_boxes[:, 0])),  # x1
                    int(np.median(cow_boxes[:, 1])),  # y1
                    int(np.median(cow_boxes[:, 2])),  # x2
                    int(np.median(cow_boxes[:, 3]))   # y2
                ]
                # Add padding
                padding = 50
                crop_box[0] = max(0, crop_box[0] - padding)
                crop_box[1] = max(0, crop_box[1] - padding)
                crop_box[2] = min(width, crop_box[2] + padding)
                crop_box[3] = min(height, crop_box[3] + padding)
            
            # Crop video using moviepy
            video = VideoFileClip(str(input_path))
            cropped = video.crop(
                x1=crop_box[0],
                y1=crop_box[1],
                x2=crop_box[2],
                y2=crop_box[3]
            )
            
            # Write cropped video
            cropped.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                fps=fps
            )
            
            video.close()
            cropped.close()
            
            print(f"Cropped video saved to {output_path}")
            
            # Publish preprocessed event
            preprocessed_data = {
                "video_id": video_id,
                "original_path": str(input_path),
                "processed_path": str(output_path),
                "crop_box": crop_box,
                "fps": fps,
                "width": crop_box[2] - crop_box[0],
                "height": crop_box[3] - crop_box[1],
                "total_frames": total_frames
            }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["video_preprocessed"],
                preprocessed_data
            )
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the preprocessing service"""
        await self.nats_client.connect()
        
        # Subscribe to video.uploaded events
        subject = self.config["nats"]["subjects"]["video_uploaded"]
        print(f"Subscribing to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("Video preprocessing service started. Waiting for videos...")
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    preprocessor = VideoPreprocessor()
    await preprocessor.start()


if __name__ == "__main__":
    asyncio.run(main())

