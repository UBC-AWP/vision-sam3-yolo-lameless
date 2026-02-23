#!/usr/bin/env python3
"""
Convert uploaded videos from FMP4/MPEG4 to H.264 codec for browser compatibility.

This script converts videos that are already uploaded in data/videos/
and replaces them in-place (with backup).

Usage:
    python scripts/convert_uploaded_videos.py                    # Convert all non-H.264 videos
    python scripts/convert_uploaded_videos.py --dry-run          # Preview only
    python scripts/convert_uploaded_videos.py --limit 5          # Convert first 5 videos
"""

import subprocess
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Configuration - use Docker volume path
PROJECT_ROOT = Path(__file__).parent.parent
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
BACKUP_DIR = PROJECT_ROOT / "data" / "videos_backup_mpeg4"


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_video_codec(path: Path) -> str:
    """Get video codec using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                str(path)
            ],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def convert_video(input_path: Path, output_path: Path) -> bool:
    """Convert video to H.264 codec."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path)
            ],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def print_progress(current: int, total: int, filename: str, status: str = ""):
    """Print progress bar."""
    pct = (current / total) * 100 if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_len - filled)

    line = f"\r[{bar}] {current}/{total} ({pct:5.1f}%) {filename[:30]:30}"
    if status:
        line += f" {status}"

    sys.stdout.write(line)
    sys.stdout.flush()


def main(args):
    """Main conversion logic."""
    start_time = datetime.now()

    # Check ffmpeg
    if not check_ffmpeg():
        print("ERROR: ffmpeg is not installed or not in PATH")
        print("Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
        return

    # Find videos to convert
    if not VIDEOS_DIR.exists():
        print(f"ERROR: Videos directory not found: {VIDEOS_DIR}")
        return

    all_videos = sorted(VIDEOS_DIR.glob("*.mp4"))

    # Filter to only non-H.264 videos
    videos_to_convert = []
    print("Scanning videos for codec...")
    for i, video in enumerate(all_videos):
        codec = get_video_codec(video)
        if codec not in ["h264", "unknown"]:
            videos_to_convert.append((video, codec))
        if (i + 1) % 50 == 0:
            print(f"  Scanned {i + 1}/{len(all_videos)}...")

    if args.skip > 0:
        videos_to_convert = videos_to_convert[args.skip:]

    if args.limit > 0:
        videos_to_convert = videos_to_convert[:args.limit]

    print(f"\n{'=' * 50}")
    print("UPLOADED VIDEO CONVERSION (MPEG4 -> H.264)")
    print(f"{'=' * 50}")
    print(f"Source: {VIDEOS_DIR}")
    print(f"Backup: {BACKUP_DIR}")
    print(f"Total videos: {len(all_videos)}")
    print(f"Need conversion: {len(videos_to_convert)}")

    if args.skip > 0:
        print(f"Skipping first: {args.skip}")

    print(f"{'=' * 50}\n")

    if args.dry_run:
        print("DRY RUN - No files will be converted\n")
        for i, (video, codec) in enumerate(videos_to_convert[:20]):
            print(f"  {video.name}: {codec}")
        if len(videos_to_convert) > 20:
            print(f"  ... and {len(videos_to_convert) - 20} more")
        return

    if not videos_to_convert:
        print("No videos need conversion! All videos are already H.264.")
        return

    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Convert videos
    converted = 0
    failed = 0

    for i, (video, codec) in enumerate(videos_to_convert):
        # Create temp output path
        temp_output = video.with_suffix(".h264.mp4")
        backup_path = BACKUP_DIR / video.name

        print_progress(i + 1, len(videos_to_convert), video.name, f"converting ({codec})...")

        if convert_video(video, temp_output):
            # Backup original
            shutil.move(str(video), str(backup_path))
            # Replace with converted
            shutil.move(str(temp_output), str(video))
            converted += 1
            print_progress(i + 1, len(videos_to_convert), video.name, "done")
        else:
            failed += 1
            # Clean up temp file if exists
            if temp_output.exists():
                temp_output.unlink()
            print_progress(i + 1, len(videos_to_convert), video.name, "FAILED")

    print("\n")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Converted: {converted}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f} seconds")
    print(f"{'=' * 50}")

    if converted > 0:
        print(f"\nOriginal files backed up to: {BACKUP_DIR}")
        print(f"\nVideos in {VIDEOS_DIR} are now H.264 compatible!")
        print("Refresh your browser to see the changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert uploaded videos from MPEG4 to H.264 codec"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without converting"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        help="Limit number of videos to convert (0 = all)"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N videos"
    )

    args = parser.parse_args()
    main(args)
