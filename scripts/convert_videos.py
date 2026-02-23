#!/usr/bin/env python3
"""
Convert videos from FMP4 to H.264 codec for browser compatibility.

Usage:
    python scripts/convert_videos.py                    # Convert all UBC videos
    python scripts/convert_videos.py --limit 5          # Convert first 5 videos
    python scripts/convert_videos.py --dry-run          # Preview only
"""

import subprocess
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
UBC_DIR = PROJECT_ROOT / "research" / "UBC"
OUTPUT_DIR = PROJECT_ROOT / "research" / "UBC_h264"


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
    if not UBC_DIR.exists():
        print(f"ERROR: UBC directory not found: {UBC_DIR}")
        return

    videos = sorted(UBC_DIR.glob("*.mp4"))

    if args.skip > 0:
        videos = videos[args.skip:]

    if args.limit > 0:
        videos = videos[:args.limit]

    print(f"\n{'=' * 50}")
    print("VIDEO CONVERSION (FMP4 -> H.264)")
    print(f"{'=' * 50}")
    print(f"Source: {UBC_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Videos to convert: {len(videos)}")

    if args.skip > 0:
        print(f"Skipping first: {args.skip}")

    print(f"{'=' * 50}\n")

    if args.dry_run:
        print("DRY RUN - No files will be converted\n")
        for i, video in enumerate(videos[:10]):
            codec = get_video_codec(video)
            print(f"  {video.name}: {codec}")
        if len(videos) > 10:
            print(f"  ... and {len(videos) - 10} more")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert videos
    converted = 0
    skipped = 0
    failed = 0

    for i, video in enumerate(videos):
        output_path = OUTPUT_DIR / video.name

        # Skip if already converted
        if output_path.exists() and output_path.stat().st_size > 0:
            print_progress(i + 1, len(videos), video.name, "skipped (exists)")
            skipped += 1
            continue

        # Check codec
        codec = get_video_codec(video)
        if codec == "h264":
            # Already H.264, just copy
            print_progress(i + 1, len(videos), video.name, "copying (already h264)")
            import shutil
            shutil.copy2(video, output_path)
            converted += 1
            continue

        # Convert
        print_progress(i + 1, len(videos), video.name, "converting...")

        if convert_video(video, output_path):
            converted += 1
            print_progress(i + 1, len(videos), video.name, "done")
        else:
            failed += 1
            print_progress(i + 1, len(videos), video.name, "FAILED")

    print("\n")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f} seconds")
    print(f"{'=' * 50}")

    if converted > 0:
        print(f"\nConverted videos are in: {OUTPUT_DIR}")
        print(f"\nTo upload converted videos, update the UBC_DIR in bulk_upload.py")
        print(f"or create a symlink: ln -sf {OUTPUT_DIR} {UBC_DIR}_backup && mv {UBC_DIR} {UBC_DIR}_original && ln -sf {OUTPUT_DIR} {UBC_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert UBC videos from FMP4 to H.264 codec"
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
