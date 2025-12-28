"""
Clip Curation Service
Converts raw video uploads into canonical 5-second clips optimized for lameness assessment.

Key Features:
- Transcode to fixed FPS/resolution
- Walking segment detection via centroid tracking
- Direction normalization (flip right→left to canonical left→right)
- 5s window scoring based on quality metrics
- Output: canonical_5s_clip.mp4 + quality_report.json
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from ultralytics import YOLO
import yaml
from shared.utils.nats_client import NATSClient


@dataclass
class QualityMetrics:
    """Quality metrics for a video window"""
    framing_score: float  # 0-1, bbox size and position
    steadiness_score: float  # 0-1, walking speed consistency
    straightness_score: float  # 0-1, how straight the cow walks
    visual_quality_score: float  # 0-1, blur and brightness
    occlusion_score: float  # 0-1, visibility (1 = no occlusion)
    overall_score: float  # Weighted combination
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass  
class WalkingPass:
    """Represents a single walking pass through the frame"""
    start_frame: int
    end_frame: int
    direction: str  # 'left_to_right' or 'right_to_left'
    centroids: List[Tuple[float, float]]
    bboxes: List[List[float]]
    confidences: List[float]
    normalized_progress: List[float]  # 0-1 progress through pass


@dataclass
class ClipCandidate:
    """A candidate 5s window for the canonical clip"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    metrics: QualityMetrics
    needs_flip: bool


class ClipCurator:
    """
    Curates canonical 5-second clips from raw video uploads.
    
    Process:
    1. Detect cow and track across frames
    2. Identify walking passes (direction changes)
    3. Normalize direction to left→right
    4. Score candidate 5s windows
    5. Select best window and export
    """
    
    # Configuration
    TARGET_FPS = 25
    TARGET_RESOLUTION = (1280, 720)
    CANONICAL_DURATION = 5.0  # seconds
    MIN_PASS_FRAMES = 30  # Minimum frames for a valid walking pass
    PROGRESS_BAND = (0.25, 0.85)  # Valid progress range for clip selection
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Initialize YOLO for cow detection
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Directories
        self.videos_dir = Path("/app/data/videos")
        self.canonical_dir = Path("/app/data/canonical")
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = Path("/app/data/quality_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def detect_cow_in_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect cow in a single frame using YOLO"""
        results = self.yolo_model(frame, verbose=False, conf=0.3)
        
        best_detection = None
        best_area = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Class 19 is cow in COCO, or accept any large detection
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                
                # Accept if it's a cow or a large detection (>10% of frame)
                if (cls == 19 or area > frame_area * 0.1) and area > best_area:
                    best_area = area
                    best_detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'centroid': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'area': area
                    }
        
        return best_detection
    
    def track_cow_through_video(self, video_path: Path) -> Tuple[List[Dict], Dict]:
        """Track cow through entire video, extracting detections per frame"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_info = {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'duration': total_frames / fps if fps > 0 else 0
        }
        
        detections = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detection = self.detect_cow_in_frame(frame)
            detections.append({
                'frame': frame_idx,
                'time': frame_idx / fps if fps > 0 else 0,
                'detection': detection
            })
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Tracked {frame_idx}/{total_frames} frames")
        
        cap.release()
        return detections, video_info
    
    def identify_walking_passes(self, detections: List[Dict], video_info: Dict) -> List[WalkingPass]:
        """Identify distinct walking passes based on direction changes"""
        passes = []
        current_pass_start = None
        current_direction = None
        current_centroids = []
        current_bboxes = []
        current_confidences = []
        
        width = video_info['width']
        
        for det in detections:
            if det['detection'] is None:
                # Gap in detection - end current pass if exists
                if current_pass_start is not None and len(current_centroids) >= self.MIN_PASS_FRAMES:
                    passes.append(self._create_walking_pass(
                        current_pass_start, det['frame'] - 1,
                        current_direction, current_centroids, 
                        current_bboxes, current_confidences, width
                    ))
                current_pass_start = None
                current_centroids = []
                current_bboxes = []
                current_confidences = []
                continue
            
            centroid = det['detection']['centroid']
            bbox = det['detection']['bbox']
            conf = det['detection']['confidence']
            
            if current_pass_start is None:
                # Start new pass
                current_pass_start = det['frame']
                current_centroids = [centroid]
                current_bboxes = [bbox]
                current_confidences = [conf]
            else:
                # Check for direction change
                if len(current_centroids) >= 5:
                    recent_x = [c[0] for c in current_centroids[-5:]]
                    x_movement = centroid[0] - recent_x[0]
                    
                    new_direction = 'left_to_right' if x_movement > 0 else 'right_to_left'
                    
                    if current_direction is None:
                        current_direction = new_direction
                    elif new_direction != current_direction and abs(x_movement) > width * 0.05:
                        # Direction changed - save current pass and start new
                        if len(current_centroids) >= self.MIN_PASS_FRAMES:
                            passes.append(self._create_walking_pass(
                                current_pass_start, det['frame'] - 1,
                                current_direction, current_centroids,
                                current_bboxes, current_confidences, width
                            ))
                        
                        current_pass_start = det['frame']
                        current_direction = new_direction
                        current_centroids = [centroid]
                        current_bboxes = [bbox]
                        current_confidences = [conf]
                        continue
                
                current_centroids.append(centroid)
                current_bboxes.append(bbox)
                current_confidences.append(conf)
        
        # Save final pass
        if current_pass_start is not None and len(current_centroids) >= self.MIN_PASS_FRAMES:
            passes.append(self._create_walking_pass(
                current_pass_start, detections[-1]['frame'],
                current_direction or 'left_to_right', current_centroids,
                current_bboxes, current_confidences, width
            ))
        
        return passes
    
    def _create_walking_pass(self, start_frame: int, end_frame: int, 
                             direction: str, centroids: List[Tuple[float, float]],
                             bboxes: List[List[float]], confidences: List[float],
                             frame_width: float) -> WalkingPass:
        """Create a WalkingPass object with normalized progress"""
        # Calculate normalized progress based on x position
        x_positions = [c[0] for c in centroids]
        min_x, max_x = min(x_positions), max(x_positions)
        x_range = max_x - min_x if max_x > min_x else 1
        
        if direction == 'left_to_right':
            normalized_progress = [(x - min_x) / x_range for x in x_positions]
        else:
            normalized_progress = [(max_x - x) / x_range for x in x_positions]
        
        return WalkingPass(
            start_frame=start_frame,
            end_frame=end_frame,
            direction=direction,
            centroids=centroids,
            bboxes=bboxes,
            confidences=confidences,
            normalized_progress=normalized_progress
        )
    
    def compute_blur_score(self, frame: np.ndarray) -> float:
        """Compute blur score using Laplacian variance (higher = sharper)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 (assuming typical range 0-500)
        return min(1.0, laplacian_var / 500.0)
    
    def compute_brightness_score(self, frame: np.ndarray) -> float:
        """Compute brightness quality (penalize too dark or too bright)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_brightness = np.mean(gray)
        # Optimal brightness around 128, penalize extremes
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        return max(0.0, brightness_score)
    
    def score_window(self, walking_pass: WalkingPass, start_idx: int, 
                     window_frames: int, video_info: Dict,
                     cap: cv2.VideoCapture) -> QualityMetrics:
        """Score a candidate window within a walking pass"""
        end_idx = min(start_idx + window_frames, len(walking_pass.centroids))
        
        if end_idx - start_idx < window_frames * 0.8:
            # Window too short
            return QualityMetrics(0, 0, 0, 0, 0, 0)
        
        window_centroids = walking_pass.centroids[start_idx:end_idx]
        window_bboxes = walking_pass.bboxes[start_idx:end_idx]
        window_confidences = walking_pass.confidences[start_idx:end_idx]
        window_progress = walking_pass.normalized_progress[start_idx:end_idx]
        
        frame_width = video_info['width']
        frame_height = video_info['height']
        
        # 1. Framing Score: bbox size and position (not touching edges)
        avg_bbox_area = np.mean([
            (b[2] - b[0]) * (b[3] - b[1]) for b in window_bboxes
        ])
        frame_area = frame_width * frame_height
        relative_size = avg_bbox_area / frame_area
        size_score = min(1.0, relative_size / 0.3)  # Optimal if cow fills ~30% of frame
        
        # Check edge proximity
        edge_penalties = []
        for bbox in window_bboxes:
            x1, y1, x2, y2 = bbox
            left_margin = x1 / frame_width
            right_margin = (frame_width - x2) / frame_width
            top_margin = y1 / frame_height
            bottom_margin = (frame_height - y2) / frame_height
            min_margin = min(left_margin, right_margin, top_margin, bottom_margin)
            edge_penalties.append(min(1.0, min_margin / 0.05))  # Penalize if < 5% margin
        
        framing_score = size_score * 0.6 + np.mean(edge_penalties) * 0.4
        
        # 2. Steadiness Score: consistent walking speed
        x_positions = [c[0] for c in window_centroids]
        x_velocities = np.diff(x_positions)
        speed_mean = np.abs(np.mean(x_velocities))
        speed_std = np.std(x_velocities)
        
        # Penalize high variability relative to mean
        if speed_mean > 0:
            cv = speed_std / speed_mean  # Coefficient of variation
            steadiness_score = max(0, 1.0 - cv)
        else:
            steadiness_score = 0.0
        
        # 3. Straightness Score: how straight the cow walks (minimal y variation)
        y_positions = [c[1] for c in window_centroids]
        y_std = np.std(y_positions)
        y_range = max(y_positions) - min(y_positions)
        # Normalize by frame height
        straightness_score = max(0, 1.0 - (y_range / frame_height) * 10)
        
        # 4. Visual Quality: sample frames for blur and brightness
        visual_scores = []
        sample_frames_idx = [
            walking_pass.start_frame + start_idx + i 
            for i in range(0, end_idx - start_idx, max(1, (end_idx - start_idx) // 5))
        ]
        
        for frame_idx in sample_frames_idx[:5]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                blur_score = self.compute_blur_score(frame)
                brightness_score = self.compute_brightness_score(frame)
                visual_scores.append((blur_score + brightness_score) / 2)
        
        visual_quality_score = np.mean(visual_scores) if visual_scores else 0.5
        
        # 5. Occlusion Score: based on detection confidence
        occlusion_score = np.mean(window_confidences)
        
        # 6. Progress band check: prefer mid-pass segments
        avg_progress = np.mean(window_progress)
        progress_penalty = 1.0
        if avg_progress < self.PROGRESS_BAND[0]:
            progress_penalty = avg_progress / self.PROGRESS_BAND[0]
        elif avg_progress > self.PROGRESS_BAND[1]:
            progress_penalty = (1.0 - avg_progress) / (1.0 - self.PROGRESS_BAND[1])
        
        # Overall weighted score
        overall_score = (
            framing_score * 0.25 +
            steadiness_score * 0.25 +
            straightness_score * 0.15 +
            visual_quality_score * 0.15 +
            occlusion_score * 0.1 +
            progress_penalty * 0.1
        )
        
        return QualityMetrics(
            framing_score=float(framing_score),
            steadiness_score=float(steadiness_score),
            straightness_score=float(straightness_score),
            visual_quality_score=float(visual_quality_score),
            occlusion_score=float(occlusion_score),
            overall_score=float(overall_score)
        )
    
    def find_best_window(self, walking_pass: WalkingPass, video_info: Dict,
                         video_path: Path) -> Optional[ClipCandidate]:
        """Find the best 5s window within a walking pass"""
        fps = video_info['fps']
        window_frames = int(self.CANONICAL_DURATION * fps)
        
        if len(walking_pass.centroids) < window_frames:
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        
        best_candidate = None
        best_score = -1
        
        # Slide window through the pass
        step = max(1, window_frames // 4)  # 25% overlap
        
        for start_idx in range(0, len(walking_pass.centroids) - window_frames + 1, step):
            metrics = self.score_window(walking_pass, start_idx, window_frames, video_info, cap)
            
            if metrics.overall_score > best_score:
                best_score = metrics.overall_score
                start_frame = walking_pass.start_frame + start_idx
                end_frame = start_frame + window_frames
                
                best_candidate = ClipCandidate(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    metrics=metrics,
                    needs_flip=(walking_pass.direction == 'right_to_left')
                )
        
        cap.release()
        return best_candidate
    
    def extract_canonical_clip(self, video_path: Path, candidate: ClipCandidate,
                               output_path: Path, video_info: Dict) -> bool:
        """Extract and process the canonical 5s clip"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = self.TARGET_RESOLUTION[0]
        out_height = self.TARGET_RESOLUTION[1]
        
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.TARGET_FPS,
            (out_width, out_height)
        )
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.start_frame)
        
        frames_written = 0
        target_frames = int(self.CANONICAL_DURATION * self.TARGET_FPS)
        
        # Calculate frame sampling to match target FPS
        source_fps = video_info['fps']
        frame_ratio = source_fps / self.TARGET_FPS
        
        frame_idx = 0
        while frames_written < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames to match target FPS
            if frame_idx >= frames_written * frame_ratio:
                # Resize to target resolution
                frame = cv2.resize(frame, (out_width, out_height))
                
                # Flip if needed (normalize to left→right)
                if candidate.needs_flip:
                    frame = cv2.flip(frame, 1)
                
                out.write(frame)
                frames_written += 1
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Re-encode with ffmpeg for better compression (if available)
        try:
            import subprocess
            temp_path = output_path.with_suffix('.temp.mp4')
            output_path.rename(temp_path)
            
            subprocess.run([
                'ffmpeg', '-y', '-i', str(temp_path),
                '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                str(output_path)
            ], capture_output=True, check=True)
            
            temp_path.unlink()
        except Exception:
            # FFmpeg not available or failed, keep original
            if output_path.with_suffix('.temp.mp4').exists():
                output_path.with_suffix('.temp.mp4').rename(output_path)
        
        return frames_written > 0
    
    def generate_quality_report(self, video_id: str, video_info: Dict,
                                passes: List[WalkingPass], 
                                selected_candidate: Optional[ClipCandidate],
                                backup_candidate: Optional[ClipCandidate]) -> Dict:
        """Generate quality report for the curation process"""
        report = {
            'video_id': video_id,
            'source_video': {
                'fps': video_info['fps'],
                'width': video_info['width'],
                'height': video_info['height'],
                'total_frames': video_info['total_frames'],
                'duration': video_info['duration']
            },
            'canonical_clip': {
                'target_fps': self.TARGET_FPS,
                'target_resolution': list(self.TARGET_RESOLUTION),
                'target_duration': self.CANONICAL_DURATION
            },
            'walking_passes_detected': len(passes),
            'passes': [
                {
                    'start_frame': p.start_frame,
                    'end_frame': p.end_frame,
                    'direction': p.direction,
                    'duration': (p.end_frame - p.start_frame) / video_info['fps']
                }
                for p in passes
            ],
            'selected_window': None,
            'backup_window': None,
            'status': 'failed',
            'rejection_reason': None
        }
        
        if selected_candidate:
            report['selected_window'] = {
                'start_frame': selected_candidate.start_frame,
                'end_frame': selected_candidate.end_frame,
                'start_time': selected_candidate.start_time,
                'end_time': selected_candidate.end_time,
                'needs_flip': selected_candidate.needs_flip,
                'metrics': selected_candidate.metrics.to_dict()
            }
            report['status'] = 'success'
        else:
            report['rejection_reason'] = 'No valid walking pass found with sufficient quality'
        
        if backup_candidate:
            report['backup_window'] = {
                'start_frame': backup_candidate.start_frame,
                'end_frame': backup_candidate.end_frame,
                'start_time': backup_candidate.start_time,
                'end_time': backup_candidate.end_time,
                'needs_flip': backup_candidate.needs_flip,
                'metrics': backup_candidate.metrics.to_dict()
            }
        
        return report
    
    async def curate_video(self, video_data: dict):
        """Main entry point: curate a video into canonical 5s clip"""
        video_id = video_data.get("video_id")
        input_path = Path(video_data.get("file_path", ""))
        
        if not input_path.exists():
            # Try to find in videos directory
            video_files = list(self.videos_dir.glob(f"{video_id}.*"))
            if video_files:
                input_path = video_files[0]
            else:
                print(f"Video not found: {video_id}")
                return
        
        print(f"=" * 60)
        print(f"Curating video: {video_id}")
        print(f"Input: {input_path}")
        print(f"=" * 60)
        
        try:
            # Step 1: Track cow through video
            print("Step 1: Tracking cow through video...")
            detections, video_info = self.track_cow_through_video(input_path)
            print(f"  Tracked {len(detections)} frames")
            
            # Step 2: Identify walking passes
            print("Step 2: Identifying walking passes...")
            passes = self.identify_walking_passes(detections, video_info)
            print(f"  Found {len(passes)} walking passes")
            
            for i, p in enumerate(passes):
                duration = (p.end_frame - p.start_frame) / video_info['fps']
                print(f"    Pass {i+1}: {p.direction}, {duration:.1f}s")
            
            # Step 3: Find best windows in each pass
            print("Step 3: Scoring candidate windows...")
            candidates = []
            for walking_pass in passes:
                candidate = self.find_best_window(walking_pass, video_info, input_path)
                if candidate:
                    candidates.append(candidate)
                    print(f"    Window found: score={candidate.metrics.overall_score:.3f}")
            
            # Sort by score and select best + backup
            candidates.sort(key=lambda c: c.metrics.overall_score, reverse=True)
            selected = candidates[0] if candidates else None
            backup = candidates[1] if len(candidates) > 1 else None
            
            # Step 4: Extract canonical clip
            output_path = self.canonical_dir / f"{video_id}_canonical.mp4"
            backup_path = self.canonical_dir / f"{video_id}_backup.mp4"
            
            if selected:
                print("Step 4: Extracting canonical clip...")
                success = self.extract_canonical_clip(input_path, selected, output_path, video_info)
                
                if success:
                    print(f"  ✅ Canonical clip saved: {output_path}")
                    
                    if backup:
                        self.extract_canonical_clip(input_path, backup, backup_path, video_info)
                        print(f"  ✅ Backup clip saved: {backup_path}")
            else:
                print("  ❌ No valid window found")
            
            # Step 5: Generate quality report
            print("Step 5: Generating quality report...")
            report = self.generate_quality_report(video_id, video_info, passes, selected, backup)
            
            report_path = self.reports_dir / f"{video_id}_quality.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"  Report saved: {report_path}")
            
            # Publish curated event
            curated_data = {
                'video_id': video_id,
                'status': report['status'],
                'canonical_path': str(output_path) if selected else None,
                'backup_path': str(backup_path) if backup else None,
                'report_path': str(report_path),
                'quality_score': selected.metrics.overall_score if selected else 0,
                'needs_flip': selected.needs_flip if selected else False,
                'source_duration': video_info['duration'],
                'canonical_duration': self.CANONICAL_DURATION
            }
            
            await self.nats_client.publish(
                self.config.get("nats", {}).get("subjects", {}).get("video_curated", "video.curated"),
                curated_data
            )
            
            print(f"=" * 60)
            print(f"Curation complete for {video_id}")
            print(f"Status: {report['status']}")
            if selected:
                print(f"Quality score: {selected.metrics.overall_score:.3f}")
            print(f"=" * 60)
            
        except Exception as e:
            print(f"❌ Error curating video {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the clip curation service"""
        await self.nats_client.connect()
        
        # Subscribe to video uploaded events (or preprocessed)
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "video_uploaded", "video.uploaded"
        )
        print(f"Clip Curation Service subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.curate_video)
        
        print("=" * 60)
        print("Clip Curation Service Started")
        print("=" * 60)
        print(f"Target FPS: {self.TARGET_FPS}")
        print(f"Target Resolution: {self.TARGET_RESOLUTION}")
        print(f"Canonical Duration: {self.CANONICAL_DURATION}s")
        print(f"Progress Band: {self.PROGRESS_BAND}")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    curator = ClipCurator()
    await curator.start()


if __name__ == "__main__":
    asyncio.run(main())

