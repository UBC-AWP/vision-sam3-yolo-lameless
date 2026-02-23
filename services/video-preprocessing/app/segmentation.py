from __future__ import annotations

import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


COW_CONF_THRESHOLD = 0.3
DEFAULT_FPS = 30

MAJ_WINDOW_FRAMES = 15
GAP_FILL_SECONDS = 1.0
MIN_RUN_SECONDS = 0.5
MIN_ON_SECONDS = 0
PAD_SECONDS = 0.5
MERGE_NEIGHBOR_SECONDS = 0.1

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg",
)


@dataclass
class Segment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def read_yolo_cow_presence(csv_path: Path) -> np.ndarray:
    max_conf_by_frame: Dict[int, float] = {}
    max_frame = -1
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                frame_idx = int(row[0])
            except ValueError:
                continue
            max_frame = max(max_frame, frame_idx)
            cls = row[1].strip().lower()
            if cls != "cow":
                continue
            try:
                conf = float(row[2])
            except ValueError:
                continue
            prev = max_conf_by_frame.get(frame_idx)
            if prev is None or conf > prev:
                max_conf_by_frame[frame_idx] = conf

    if max_frame < 0:
        return np.zeros(0, dtype=np.uint8)

    presence = np.zeros(max_frame + 1, dtype=np.uint8)
    for frame_idx, conf in max_conf_by_frame.items():
        if conf >= COW_CONF_THRESHOLD:
            presence[frame_idx] = 1
    return presence


def majority_filter(binary: np.ndarray, window_frames: int) -> np.ndarray:
    if binary.size == 0 or window_frames <= 1:
        return binary.copy()
    kernel = np.ones(window_frames, dtype=np.int32)
    counts = np.convolve(binary.astype(np.int32), kernel, mode="same")
    threshold = int(math.ceil(window_frames / 2.0))
    return (counts >= threshold).astype(np.uint8)


def fill_short_gaps(binary: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if binary.size == 0 or max_gap_frames <= 0:
        return binary.copy()
    arr = binary.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 1:
            i += 1
            continue
        start = i
        while i < n and arr[i] == 0:
            i += 1
        end = i - 1
        left_on = start > 0 and arr[start - 1] == 1
        right_on = i < n and arr[i] == 1
        gap_len = end - start + 1
        if left_on and right_on and gap_len <= max_gap_frames:
            arr[start : end + 1] = 1
    return arr


def remove_short_runs(binary: np.ndarray, min_run_frames: int) -> np.ndarray:
    if binary.size == 0 or min_run_frames <= 1:
        return binary.copy()
    arr = binary.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 0:
            i += 1
            continue
        start = i
        while i < n and arr[i] == 1:
            i += 1
        end = i - 1
        if end - start + 1 < min_run_frames:
            arr[start : end + 1] = 0
    return arr


def extract_intervals(binary: np.ndarray) -> List[Segment]:
    segments: List[Segment] = []
    n = len(binary)
    i = 0
    while i < n:
        if binary[i] == 0:
            i += 1
            continue
        start = i
        while i < n and binary[i] == 1:
            i += 1
        end = i - 1
        segments.append(Segment(start, end))
    return segments


def merge_segments_with_gap(segments: List[Segment], max_gap: int) -> List[Segment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    merged: List[Segment] = []
    cur = segments[0]
    for seg in segments[1:]:
        gap = seg.start - cur.end - 1
        if gap <= max_gap:
            cur = Segment(cur.start, max(cur.end, seg.end))
        else:
            merged.append(cur)
            cur = seg
    merged.append(cur)
    return merged


def get_video_fps(video_path: Path) -> Optional[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def build_segments_for_video(csv_path: Path, video_path: Path) -> List[Dict[str, object]]:
    presence_raw = read_yolo_cow_presence(csv_path)
    if presence_raw.size == 0:
        return []

    fps = get_video_fps(video_path)
    if fps is None or fps <= 0:
        fps = float(DEFAULT_FPS)
        print(f"[WARN] Using DEFAULT_FPS={DEFAULT_FPS} for {video_path}")

    pad_frames = int(round(fps * PAD_SECONDS))
    merge_neighbor_frames = int(round(fps * MERGE_NEIGHBOR_SECONDS))
    min_on_frames = int(round(fps * MIN_ON_SECONDS))
    gap_fill_frames = int(round(fps * GAP_FILL_SECONDS))
    min_run_frames = int(round(fps * MIN_RUN_SECONDS))

    presence = majority_filter(presence_raw, MAJ_WINDOW_FRAMES)
    presence = fill_short_gaps(presence, gap_fill_frames)
    presence = remove_short_runs(presence, min_run_frames)

    intervals = extract_intervals(presence)
    padded = [Segment(max(0, s.start - pad_frames), s.end + pad_frames) for s in intervals]
    merged = merge_segments_with_gap(padded, merge_neighbor_frames)
    valid_segments = [s for s in merged if s.length >= min_on_frames]

    rows: List[Dict[str, object]] = []
    for seq, seg_item in enumerate(valid_segments, start=1):
        rows.append(
            {
                "date": "manual",
                "long_video": video_path.stem,
                "short_seq": seq,
                "frames": seg_item.length,
                "fps": fps,
                "start_frame": seg_item.start,
                "end_frame": seg_item.end,
                "start_sec": seg_item.start / fps,
                "end_sec": (seg_item.end + 1) / fps,
                "_video_path": str(video_path),
            }
        )
    return rows


def write_metadata(rows: List[Dict[str, object]], output_csv: Path) -> None:
    fieldnames = [
        "date",
        "long_video",
        "short_seq",
        "frames",
        "fps",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: v for k, v in row.items() if k in fieldnames} for row in rows])


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH; please install ffmpeg or add to PATH.")


def cut_segments_from_rows(rows: List[Dict[str, object]], output_dir: Path, log_csv: Path) -> None:
    ensure_ffmpeg_available()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_fields = [
        "long_video",
        "short_seq",
        "output_file",
        "status",
        "message",
        "start_sec",
        "end_sec",
    ]
    log_rows: List[Dict[str, object]] = []

    for row in rows:
        long_video = row.get("long_video")
        seq = row.get("short_seq")
        video_path_str = row.get("_video_path", "")
        if not long_video or seq is None or not video_path_str:
            log_rows.append(
                {
                    "long_video": long_video or "",
                    "short_seq": seq if seq is not None else "",
                    "output_file": "",
                    "status": "skip",
                    "message": "missing video path or identifiers",
                    "start_sec": row.get("start_sec", ""),
                    "end_sec": row.get("end_sec", ""),
                }
            )
            continue

        try:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
        except Exception:
            log_rows.append(
                {
                    "long_video": long_video,
                    "short_seq": seq,
                    "output_file": "",
                    "status": "error",
                    "message": "invalid start/end seconds",
                    "start_sec": row.get("start_sec", ""),
                    "end_sec": row.get("end_sec", ""),
                }
            )
            continue

        duration = max(0.0, end_sec - start_sec)
        if duration <= 0:
            log_rows.append(
                {
                    "long_video": long_video,
                    "short_seq": seq,
                    "output_file": "",
                    "status": "skip",
                    "message": "non-positive duration",
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
            )
            continue

        video_path = Path(video_path_str)
        if not video_path.exists():
            log_rows.append(
                {
                    "long_video": long_video,
                    "short_seq": seq,
                    "output_file": "",
                    "status": "error",
                    "message": f"video not found: {video_path}",
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
            )
            continue

        suffix = video_path.suffix or ".mp4"
        output_file = output_dir / f"{long_video}_short_{seq}{suffix}"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True)
            status = "ok"
            message = ""
        except subprocess.CalledProcessError as e:
            status = "error"
            message = f"ffmpeg failed (code {e.returncode})"
        except Exception as e:
            status = "error"
            message = str(e)

        log_rows.append(
            {
                "long_video": long_video,
                "short_seq": seq,
                "output_file": str(output_file),
                "status": status,
                "message": message,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[CUT] Wrote {len(log_rows)} log rows to {log_csv}")
