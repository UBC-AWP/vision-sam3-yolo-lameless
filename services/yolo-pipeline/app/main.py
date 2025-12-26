"""
YOLO Detection Pipeline
Detects cows and body parts in preprocessed videos
"""
import asyncio
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from shared.utils.nats_client import NATSClient
from typing import List, Dict, Any


class YOLOPipeline:
    """YOLO detection pipeline for cow and body part detection"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Initialize YOLO model
        # TODO: Replace with custom trained cow detection model
        model_path = Path("/app/shared/models/yolo")
        if model_path.exists() and list(model_path.glob("*.pt")):
            # Use custom model if available
            model_file = list(model_path.glob("*.pt"))[0]
            self.yolo_model = YOLO(str(model_file))
            print(f"Loaded custom YOLO model: {model_file}")
        else:
            # Use pretrained model
            self.yolo_model = YOLO("yolov8n.pt")
            print("Using pretrained YOLOv8n model")
        
        self.confidence_threshold = self.config.get("models", {}).get("yolo", {}).get("confidence_threshold", 0.5)
        
        # Directories
        self.processed_dir = Path("/app/data/processed")
        self.results_dir = Path("/app/data/results/yolo")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def detect_in_video(self, video_path: Path) -> Dict[str, Any]:
        """Detect cows and body parts in video"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections = []
        frame_detections = []
        frame_count = 0
        
        # Process every Nth frame for efficiency (adjust as needed)
        frame_interval = max(1, fps // 2)  # Process 2 frames per second
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Run YOLO detection
                results = self.yolo_model(frame, verbose=False, conf=self.confidence_threshold)
                
                frame_detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.yolo_model.names[cls] if hasattr(self.yolo_model, 'names') else f"class_{cls}"
                        
                        detection = {
                            "frame": frame_count,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "class": class_name,
                            "class_id": cls
                        }
                        frame_detections.append(detection)
                
                if frame_detections:
                    detections.append({
                        "frame": frame_count,
                        "time": frame_count / fps if fps > 0 else 0,
                        "detections": frame_detections
                    })
            
            frame_count += 1
        
        cap.release()
        
        # Compute aggregated features
        features = self._compute_features(detections, total_frames, fps)
        
        return {
            "detections": detections,
            "features": features,
            "total_frames": total_frames,
            "fps": fps,
            "frames_processed": len(detections)
        }
    
    def _compute_features(self, detections: List[Dict], total_frames: int, fps: float) -> Dict[str, Any]:
        """Compute detection features"""
        if not detections:
            return {}
        
        # Extract all bounding boxes
        all_boxes = []
        confidences = []
        
        for frame_data in detections:
            for det in frame_data["detections"]:
                all_boxes.append(det["bbox"])
                confidences.append(det["confidence"])
        
        if not all_boxes:
            return {}
        
        all_boxes = np.array(all_boxes)
        confidences = np.array(confidences)
        
        # Compute box statistics
        box_widths = all_boxes[:, 2] - all_boxes[:, 0]
        box_heights = all_boxes[:, 3] - all_boxes[:, 1]
        box_areas = box_widths * box_heights
        box_centers_x = (all_boxes[:, 0] + all_boxes[:, 2]) / 2
        box_centers_y = (all_boxes[:, 1] + all_boxes[:, 3]) / 2
        
        # Compute stability (variance of box positions)
        position_stability = 1.0 / (1.0 + np.std(box_centers_x) + np.std(box_centers_y))
        
        features = {
            "num_detections": len(all_boxes),
            "avg_confidence": float(np.mean(confidences)),
            "max_confidence": float(np.max(confidences)),
            "min_confidence": float(np.min(confidences)),
            "avg_box_area": float(np.mean(box_areas)),
            "avg_box_width": float(np.mean(box_widths)),
            "avg_box_height": float(np.mean(box_heights)),
            "position_stability": float(position_stability),
            "avg_center_x": float(np.mean(box_centers_x)),
            "avg_center_y": float(np.mean(box_centers_y)),
            "detection_rate": len(detections) / total_frames if total_frames > 0 else 0
        }
        
        return features
    
    async def process_video(self, video_data: dict):
        """Process a preprocessed video"""
        video_id = video_data["video_id"]
        processed_path = Path(video_data["processed_path"])
        
        print(f"YOLO pipeline processing video {video_id}")
        
        if not processed_path.exists():
            print(f"Processed video not found: {processed_path}")
            return
        
        try:
            # Run detection
            results = self.detect_in_video(processed_path)
            
            # Save results
            results_file = self.results_dir / f"{video_id}_yolo.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish results
            pipeline_result = {
                "video_id": video_id,
                "pipeline": "yolo",
                "results_path": str(results_file),
                "features": results["features"],
                "num_detections": len(results["detections"]),
                "total_frames": results["total_frames"]
            }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["pipeline_yolo"],
                pipeline_result
            )
            
            print(f"YOLO pipeline completed for {video_id}")
            
        except Exception as e:
            print(f"Error in YOLO pipeline for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the YOLO pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to video.preprocessed events
        subject = self.config["nats"]["subjects"]["video_preprocessed"]
        print(f"YOLO pipeline subscribed to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("YOLO pipeline service started. Waiting for videos...")
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = YOLOPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

