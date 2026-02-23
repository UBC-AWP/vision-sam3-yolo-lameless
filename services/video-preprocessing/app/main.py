"""
Video Preprocessing Service (Long Video -> Good Short Clips)
Segments long videos, classifies short clips as good/bad, and publishes
good clips directly to video.preprocessed.
"""
import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd
import yaml
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import shutil
import subprocess

from shared.utils.nats_client import NATSClient
from app.detection_long_csv import load_model as load_long_model
from app.detection_long_csv import process_video_to_csv as process_long_video_to_csv
from app.detection_short_csv import load_model as load_short_model
from app.detection_short_csv import process_video_to_csv as process_short_video_to_csv
from app.feature_extraction import analyze_video_features
from app.model_utils import load_model_and_scaler, prepare_features
from app.segmentation import VIDEO_EXTENSIONS, build_segments_for_video, cut_segments_from_rows, write_metadata


class VideoPreprocessor:
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))

        self.videos_dir = Path("/app/data/videos")
        self.processed_root = Path("/app/data/processed/long_video")
        self.results_root = Path("/app/data/results/long_video")
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.models_dir = Path(os.getenv("LONG_VIDEO_MODELS_DIR", "/app/models"))

        self.long_model = None
        self.short_model = None

        # S3 upload configuration for clip videos
        self.s3_bucket = os.getenv("S3_VIDEOS_BUCKET", "")
        self.s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "")
        self.s3_force_path_style = os.getenv("S3_FORCE_PATH_STYLE", "false").lower() in {"1", "true", "yes"}
        self.s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
        self.s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.clips_videos_prefix = os.getenv("S3_CLIPS_VIDEOS_PREFIX", "clips_videos/")
        self.tenants_prefix = "tenants/"
        self._s3_client = None

    def _get_s3_client(self):
        if self._s3_client is not None or not self.s3_bucket:
            return self._s3_client
        client_kwargs = {
            "region_name": os.getenv("AWS_REGION", "us-west-2"),
        }
        if self.s3_endpoint_url:
            client_kwargs["endpoint_url"] = self.s3_endpoint_url
        if self.s3_access_key_id and self.s3_secret_access_key:
            client_kwargs["aws_access_key_id"] = self.s3_access_key_id
            client_kwargs["aws_secret_access_key"] = self.s3_secret_access_key
        if self.s3_force_path_style:
            client_kwargs["config"] = Config(s3={"addressing_style": "path"})
        self._s3_client = boto3.client("s3", **client_kwargs)
        return self._s3_client

    def _download_long_video_from_s3(self, s3_key: str, target_path: Path) -> Path | None:
        """Download a raw long video from S3 to a local path."""
        client = self._get_s3_client()
        if not client or not self.s3_bucket or not s3_key:
            return None
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(self.s3_bucket, s3_key, str(target_path))
            return target_path if target_path.exists() else None
        except Exception as e:
            print(f"[WARN] Failed to download {s3_key} from S3: {e}")
            return None

    def _upload_clips_to_s3(
        self,
        video_id: str,
        clip_videos: List[Dict[str, object]],
        tenant_id: str | None = None,
        tenant_name: str | None = None,
    ) -> Dict[str, object]:
        """Upload clip videos to S3. New key: {tenant_name}/clips/{video_id}/clip_1.mp4. Legacy: tenants/{tenant_id}/clips/ or clips_videos/{video_id}/{filename}."""
        total = len(clip_videos)
        client = self._get_s3_client()
        if not client or not self.s3_bucket:
            msg = "S3 not configured (missing bucket or client); clips not uploaded"
            print(f"[WARN] {msg}")
            return {"uploaded": 0, "failed": total, "error": msg}

        use_new_key = bool(tenant_name or tenant_id)
        uploaded = 0
        failed = 0
        last_error: str | None = None

        for idx, item in enumerate(clip_videos):
            file_path = Path(str(item.get("file_path", "")))
            if not file_path.exists():
                continue
            clip_slug = item.get("clip_slug") or f"clip_{idx + 1}"
            if use_new_key:
                if tenant_name:
                    # Use tenant_name for new path format
                    key = f"{tenant_name}/clips/{video_id}/{clip_slug}.mp4"
                else:
                    # Fallback to legacy tenants/{tenant_id}/ format
                    key = f"{self.tenants_prefix}{tenant_id}/clips/{video_id}/{clip_slug}.mp4"
            else:
                prefix = self.clips_videos_prefix
                if prefix and not prefix.endswith("/"):
                    prefix = f"{prefix}/"
                key = f"{prefix}{video_id}/{file_path.name}"
            try:
                file_size = file_path.stat().st_size
                if file_size <= 0:
                    print(f"[WARN] Skipping empty file upload: {file_path}")
                    continue
                body = file_path.read_bytes()
                # Arbutus returns MissingContentLength when boto3 put_object is used; use presigned PUT with explicit Content-Length.
                put_url = client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": self.s3_bucket, "Key": key, "ContentType": "video/mp4"},
                    ExpiresIn=3600,
                )
                req = urllib.request.Request(
                    put_url,
                    data=body,
                    method="PUT",
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(len(body)),
                    },
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    if resp.status not in (200, 204):
                        raise RuntimeError(f"PUT returned status {resp.status}")
                item["s3_key"] = key
                uploaded += 1
            except Exception as e:
                failed += 1
                last_error = str(e)
                print(f"[WARN] Failed to upload {file_path} to S3: {e}")

        return {"uploaded": uploaded, "failed": failed, "error": last_error}

    def _write_summary(
        self,
        video_id: str,
        input_path: Path | None,
        status: str,
        reason: str,
        source_fingerprint: Dict[str, object] | None = None,
        run_results_dir: Path | None = None,
        extra: Dict[str, object] | None = None,
        s3_key: str = "",
    ) -> None:
        summary_path = self.results_root / f"{video_id}_long_video.json"
        summary = {
            "video_id": video_id,
            "original_path": str(input_path) if input_path else "",
            "status": status,
            "reason": reason,
            "results_dir": str(run_results_dir) if run_results_dir else "",
        }
        if s3_key:
            summary["s3_key"] = s3_key
        if source_fingerprint is not None:
            summary["source_fingerprint"] = source_fingerprint
        if extra:
            summary.update(extra)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _write_progress(self, video_id: str, status: str, percent: int, detail: str = "") -> None:
        progress_path = self.results_root / f"{video_id}_progress.json"
        payload = {
            "video_id": video_id,
            "status": status,
            "percent": max(0, min(100, int(percent))),
            "detail": detail,
        }
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _get_source_fingerprint(self, path: Path | None) -> Dict[str, object] | None:
        if path is None or not path.exists():
            return None
        stat = path.stat()
        return {
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        }

    def _load_previous_summary(self, video_id: str) -> Dict[str, object] | None:
        summary_path = self.results_root / f"{video_id}_long_video.json"
        if not summary_path.exists():
            return None
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f) or None
        except (OSError, json.JSONDecodeError):
            return None

    def _file_nonempty(self, path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0

    def _load_metadata_rows(self, metadata_csv: Path) -> List[Dict[str, object]]:
        if not self._file_nonempty(metadata_csv):
            return []
        df = pd.read_csv(metadata_csv)
        if df.empty:
            return []
        return df.to_dict(orient="records")

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _get_long_model(self):
        if self.long_model is None:
            self.long_model = load_long_model()
        return self.long_model

    def _get_short_model(self):
        if self.short_model is None:
            self.short_model = load_short_model()
        return self.short_model

    def _compress_video(self, input_path: Path, output_path: Path) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in container")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            "scale=-2:720",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_text = result.stderr.strip() or "ffmpeg compression failed"
            raise RuntimeError(error_text)
        return output_path

    def _resolve_input_path(self, video_id: str, file_path: str | None) -> Path | None:
        if file_path:
            path = Path(file_path)
            if path.exists():
                return path
        candidates = list(self.videos_dir.glob(f"{video_id}.*"))
        return candidates[0] if candidates else None

    def _list_videos(self, root: Path) -> List[Path]:
        return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]

    def _ensure_model_files(self) -> None:
        required = ["xgboost_model.pkl", "feature_scaler.pkl", "feature_info.json"]
        missing = [name for name in required if not (self.models_dir / name).exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(
                f"Missing model files in {self.models_dir}: {missing_list}. "
                "Copy model artifacts to services/video-preprocessing/models."
            )

    async def process_video(self, video_data: dict):
        video_id = video_data.get("video_id")
        metadata = video_data.get("metadata") or {}
        s3_key = metadata.get("s3_key", "")
        tenant_id = metadata.get("tenant_id")
        if tenant_id is not None:
            tenant_id = str(tenant_id)
        # New: optional tenant_name from metadata so we can write clips under {tenant_name}/clips/...
        tenant_name = metadata.get("tenant_name")
        input_path = self._resolve_input_path(video_id, video_data.get("file_path"))
        if input_path is None and s3_key:
            filename = Path(s3_key).name or "original.mp4"
            candidate = self.videos_dir / "raw_videos" / video_id / filename
            candidate.parent.mkdir(parents=True, exist_ok=True)
            input_path = self._download_long_video_from_s3(s3_key, candidate)
        if not video_id or input_path is None:
            if video_id:
                self._write_summary(
                    video_id=video_id,
                    input_path=None,
                    status="invalid_input",
                    reason="missing or unreadable file_path",
                    source_fingerprint=None,
                    s3_key=s3_key,
                )
            print("Invalid video input; missing video_id or file path.")
            return

        original_input = input_path
        source_fingerprint = self._get_source_fingerprint(original_input)
        previous_summary = self._load_previous_summary(video_id)
        previous_fingerprint = None
        if isinstance(previous_summary, dict):
            previous_fingerprint = previous_summary.get("source_fingerprint")
        same_source = bool(source_fingerprint and previous_fingerprint == source_fingerprint)

        self._write_progress(video_id, "processing", 5, "Starting preprocessing")

        work_dir = self.processed_root / video_id
        long_csv_dir = work_dir / "long_yolo_csv"
        short_videos_dir = work_dir / "short_videos"
        short_yolo_dir = work_dir / "short_yolo_csv"
        for d in [long_csv_dir, short_videos_dir, short_yolo_dir]:
            d.mkdir(parents=True, exist_ok=True)

        run_results_dir = self.results_root / video_id
        run_results_dir.mkdir(parents=True, exist_ok=True)

        try:
            compressed_path = work_dir / f"{input_path.stem}_compressed.mp4"
            if same_source and self._file_nonempty(compressed_path):
                self._write_progress(video_id, "processing", 10, "Reusing compressed video")
                input_path = compressed_path
            else:
                self._write_progress(video_id, "processing", 10, "Compressing video")
                input_path = self._compress_video(input_path, compressed_path)

            long_csv_path = long_csv_dir / f"{input_path.stem}.csv"
            print("\n=== Step 1/4: Long video detection ===")
            if same_source and self._file_nonempty(long_csv_path):
                self._write_progress(video_id, "processing", 15, "Reusing long video detection")
            else:
                self._write_progress(video_id, "processing", 15, "Long video detection")
                ok = process_long_video_to_csv(str(input_path), str(long_csv_path), model=self._get_long_model())
                if not ok:
                    self._write_summary(
                        video_id=video_id,
                        input_path=original_input,
                        status="long_detection_failed",
                        reason="failed to process long video detections",
                        source_fingerprint=source_fingerprint,
                        run_results_dir=run_results_dir,
                        s3_key=s3_key,
                    )
                    print("Failed to process long video detections.")
                    return

            print("\n=== Step 2/4: Segment into short clips ===")
            metadata_csv = run_results_dir / "auto_short_videos_metadata.csv"
            cut_log_csv = run_results_dir / "cut_log.csv"
            rows: List[Dict[str, object]] = []
            short_videos: List[Path] = []
            reuse_segments = same_source and metadata_csv.exists() and cut_log_csv.exists()
            if reuse_segments:
                rows = self._load_metadata_rows(metadata_csv)
                short_videos = self._list_videos(short_videos_dir)
                if rows and short_videos:
                    self._write_progress(video_id, "processing", 35, "Reusing short clips")
                else:
                    reuse_segments = False

            if not reuse_segments:
                self._write_progress(video_id, "processing", 35, "Segmenting into short clips")
                rows = build_segments_for_video(long_csv_path, input_path)
                if not rows:
                    self._write_summary(
                        video_id=video_id,
                        input_path=original_input,
                        status="no_segments",
                        reason="no valid segments found",
                        source_fingerprint=source_fingerprint,
                        run_results_dir=run_results_dir,
                        extra={"good_count": 0, "bad_count": 0},
                        s3_key=s3_key,
                    )
                    print("No valid segments found; exiting.")
                    return

                write_metadata(rows, metadata_csv)
                cut_segments_from_rows(rows, short_videos_dir, cut_log_csv)
                self._write_progress(video_id, "processing", 55, "Short clips created")
                short_videos = self._list_videos(short_videos_dir)

            if not short_videos:
                self._write_summary(
                    video_id=video_id,
                    input_path=original_input,
                    status="no_short_videos",
                    reason="ffmpeg created no short clips",
                    source_fingerprint=source_fingerprint,
                    run_results_dir=run_results_dir,
                    s3_key=s3_key,
                )
                print("No short videos created; exiting.")
                return

            short_video_map = {video_path.stem: video_path for video_path in short_videos}

            print("\n=== Step 3/4: Short video detection ===")
            short_yolo_csvs = sorted(short_yolo_dir.glob("*.csv"))
            reuse_short_detection = (
                same_source and short_yolo_csvs and len(short_yolo_csvs) >= len(short_videos)
            )
            if reuse_short_detection:
                self._write_progress(video_id, "processing", 70, "Reusing short video detection")
            else:
                self._write_progress(video_id, "processing", 70, "Short video detection")
                short_model = self._get_short_model()
                for idx, video_path in enumerate(short_videos, start=1):
                    output_csv = short_yolo_dir / f"{video_path.stem}.csv"
                    print(f"[{idx}/{len(short_videos)}] {video_path.name}")
                    process_short_video_to_csv(str(video_path), str(output_csv), model=short_model)
                short_yolo_csvs = sorted(short_yolo_dir.glob("*.csv"))

            print("\n=== Step 4/4: Feature extraction + prediction ===")
            features_csv = run_results_dir / "short_video_features.csv"
            results_csv = run_results_dir / "short_video_classification_results.csv"
            reuse_results = same_source and self._file_nonempty(features_csv) and self._file_nonempty(results_csv)
            results_df = None
            threshold = None
            if reuse_results:
                self._write_progress(video_id, "processing", 85, "Reusing feature extraction and classification")
                results_df = pd.read_csv(results_csv)
                if results_df.empty:
                    reuse_results = False
                    results_df = None

            if not reuse_results:
                self._write_progress(video_id, "processing", 85, "Feature extraction and classification")
                feature_rows: List[Dict[str, object]] = []
                for csv_path in short_yolo_csvs:
                    features = analyze_video_features(str(csv_path))
                    if features is None:
                        continue
                    short_video = short_video_map.get(csv_path.stem)
                    features["file_name"] = short_video.name if short_video else f"{csv_path.stem}{csv_path.suffix}"
                    feature_rows.append(features)

                if not feature_rows:
                    self._write_summary(
                        video_id=video_id,
                        input_path=original_input,
                        status="no_features",
                        reason="no features extracted from short videos",
                        source_fingerprint=source_fingerprint,
                        run_results_dir=run_results_dir,
                        s3_key=s3_key,
                    )
                    print("No features extracted; exiting.")
                    return

                features_df = pd.DataFrame(feature_rows)
                features_df.to_csv(features_csv, index=False)

                self._ensure_model_files()
                model, scaler, feature_columns, saved_threshold = load_model_and_scaler(str(self.models_dir))
                threshold_override = os.getenv("LONG_VIDEO_BAD_THRESHOLD")
                if threshold_override:
                    try:
                        threshold = float(threshold_override)
                    except ValueError:
                        print(
                            f"[WARN] Invalid LONG_VIDEO_BAD_THRESHOLD={threshold_override}; using saved threshold."
                        )
                        threshold = saved_threshold
                else:
                    threshold = saved_threshold

                X, _ = prepare_features(features_df, feature_columns)
                X_scaled = scaler.transform(X)
                proba_bad = model.predict_proba(X_scaled)[:, 1]
                pred_bad = (proba_bad >= threshold).astype(int)

                meta_map = {f"{row['long_video']}_short_{row['short_seq']}": row for row in rows}

                output_rows = []
                for file_name, prob, pred in zip(features_df["file_name"].values, proba_bad, pred_bad):
                    stem = Path(str(file_name)).stem
                    meta = meta_map.get(stem, {})
                    label = "bad" if pred == 1 else "good"
                    output_row = {
                        "file_name": file_name,
                        "label": label,
                        "prob_bad": float(prob),
                        "start_sec": meta.get("start_sec", ""),
                        "end_sec": meta.get("end_sec", ""),
                    }
                    output_rows.append(output_row)

                results_df = pd.DataFrame(output_rows)
                results_df.to_csv(results_csv, index=False)

            if threshold is None:
                if isinstance(previous_summary, dict):
                    threshold = previous_summary.get("threshold")
                if threshold is None:
                    self._ensure_model_files()
                    _, _, _, saved_threshold = load_model_and_scaler(str(self.models_dir))
                    threshold = saved_threshold

            good_short_videos = []
            bad_short_videos = []
            clip_index = 0
            for _, row in results_df.iterrows():
                file_name = row.get("file_name")
                if not file_name:
                    continue
                stem = Path(str(file_name)).stem
                short_video = short_video_map.get(stem)
                if short_video is None:
                    continue
                clip_index += 1
                clip_slug = f"clip_{clip_index}"
                label = str(row.get("label", "")).lower()
                prob_bad = row.get("prob_bad", 0.0)
                try:
                    prob_bad = float(prob_bad)
                except (TypeError, ValueError):
                    prob_bad = 0.0
                entry = {
                    "file_name": short_video.name,
                    "file_path": str(short_video),
                    "clip_slug": clip_slug,
                    "prob_bad": prob_bad,
                    "start_sec": row.get("start_sec", ""),
                    "end_sec": row.get("end_sec", ""),
                }
                if label == "good":
                    good_short_videos.append(entry)
                else:
                    bad_short_videos.append(entry)

            self._write_progress(video_id, "processing", 95, "Uploading clips")
            all_clip_videos = good_short_videos + bad_short_videos
            upload_result = self._upload_clips_to_s3(
                video_id,
                all_clip_videos,
                tenant_id=tenant_id,
                tenant_name=tenant_name,
            )
            upload_failed_count = upload_result.get("failed", 0)
            upload_error = upload_result.get("error")
            upload_success_count = upload_result.get("uploaded", 0)
            all_upload_failed = upload_failed_count > 0 and upload_success_count == 0

            if all_upload_failed:
                self._write_progress(video_id, "upload_failed", 95, upload_error or "Upload failed")
            else:
                self._write_progress(video_id, "completed", 100, "Completed")

            s3_key = (video_data.get("metadata") or {}).get("s3_key", "")
            summary_status = "upload_failed" if all_upload_failed else "completed"
            summary_reason = (upload_error or "Clips not uploaded to S3") if all_upload_failed else ""

            summary = {
                "video_id": video_id,
                "s3_key": s3_key,
                "original_path": str(input_path),
                "status": summary_status,
                "reason": summary_reason,
                "source_fingerprint": source_fingerprint,
                "good_count": len(good_short_videos),
                "bad_count": len(bad_short_videos),
                "threshold": threshold,
                "results_dir": str(run_results_dir),
                "outputs": {
                    "metadata_csv": str(metadata_csv),
                    "cut_log_csv": str(cut_log_csv),
                    "features_csv": str(features_csv),
                    "results_csv": str(results_csv),
                },
                "good_short_videos": good_short_videos,
                "bad_short_videos": bad_short_videos,
            }
            if upload_failed_count > 0 or upload_error:
                summary["upload_failed_count"] = upload_failed_count
                summary["upload_error"] = upload_error or ""
            summary_path = self.results_root / f"{video_id}_long_video.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            subject = self.config.get("nats", {}).get("subjects", {}).get("video_preprocessed", "video.preprocessed")
            for item in good_short_videos:
                short_file_name = Path(item["file_name"]).stem
                short_video_id = short_file_name
                short_path = Path(item["file_path"])
                fps = 0
                width = 0
                height = 0
                total_frames = 0
                if short_path.exists():
                    cap = cv2.VideoCapture(str(short_path))
                    if cap.isOpened():
                        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                message = {
                    "video_id": short_video_id,
                    "file_path": item["file_path"],
                    "original_path": str(original_input),
                    "processed_path": item["file_path"],
                    "crop_box": [0, 0, width, height],
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "total_frames": total_frames,
                    "parent_video_id": video_id,
                    "start_sec": item.get("start_sec", ""),
                    "end_sec": item.get("end_sec", ""),
                    "source_pipeline": "long_video_pipeline",
                    "classification": {
                        "label": "good",
                        "prob_bad": item["prob_bad"],
                        "threshold": threshold,
                    },
                }
                if item.get("s3_key"):
                    message["s3_key"] = item["s3_key"]
                await self.nats_client.publish(subject, message)

            print(f"\nSaved results: {results_csv}")
            print(f"  Good videos: {len(good_short_videos)}")
            print(f"  Bad videos:  {len(bad_short_videos)}")
            print(f"  Threshold:   {threshold:.3f}")
        except Exception as e:
            self._write_summary(
                video_id=video_id,
                input_path=original_input,
                status="error",
                reason=str(e),
                source_fingerprint=source_fingerprint,
                run_results_dir=run_results_dir,
                s3_key=s3_key,
            )
            raise

    async def start(self):
        await self.nats_client.connect()
        subject = self.config.get("nats", {}).get("subjects", {}).get("video_uploaded", "video.uploaded")
        print(f"Subscribing to {subject}")
        await self.nats_client.subscribe(subject, self.process_video)
        print("Video preprocessing service started. Waiting for videos...")
        await asyncio.Event().wait()


async def main():
    preprocessor = VideoPreprocessor()
    await preprocessor.start()


if __name__ == "__main__":
    asyncio.run(main())

