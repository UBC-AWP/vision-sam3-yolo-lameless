#!/usr/bin/env python3
"""
Bulk Video Upload Script for Cow Lameness Detection System

Uploads training and testing videos to the backend API and triggers
the ML processing pipelines.

Training Data: research/CattleLameness/Data/
  - Lame/     : 25 videos (L (1).mp4 to L (25).mp4) - label=1
  - Normal/   : 25 videos (N (1).mp4 to N (25).mp4) - label=0

Testing Data: research/UBC/
  - 329 videos (various naming patterns) - unlabeled

Usage:
    python scripts/bulk_upload.py                    # Full upload with pipelines
    python scripts/bulk_upload.py --dry-run          # Preview files only
    python scripts/bulk_upload.py --skip-testing     # Training data only
    python scripts/bulk_upload.py --no-pipelines     # Upload without processing
"""

import asyncio
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime

import httpx

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_BASE = PROJECT_ROOT / "research" / "CattleLameness" / "Data"
LAME_DIR = TRAINING_BASE / "Lame"
NORMAL_DIR = TRAINING_BASE / "Normal"
UBC_DIR = PROJECT_ROOT / "research" / "UBC"

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_EMAIL = "admin@example.com"
DEFAULT_PASSWORD = "adminpass123"
DEFAULT_LIMIT = 0  # 0 means no limit

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
UPLOAD_TIMEOUT = 120
BATCH_SIZE = 50
DEFAULT_DELAY = 0  # seconds between uploads
PIPELINE_CHECK_INTERVAL = 5  # seconds between pipeline status checks
PIPELINE_TIMEOUT = 300  # max seconds to wait for pipeline completion


@dataclass
class UploadResult:
    """Result of a single upload"""
    file_path: str
    success: bool
    video_id: Optional[str] = None
    label: Optional[int] = None
    error: Optional[str] = None
    attempts: int = 1


@dataclass
class BatchResult:
    """Result of a batch upload"""
    category: str
    total: int = 0
    successful: int = 0
    failed: int = 0
    video_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def discover_videos(limit: int = 0, skip: int = 0) -> dict:
    """Discover all videos to upload from training and testing directories.

    Args:
        limit: Maximum videos per category (0 = no limit)
        skip: Skip first N videos in each category
    """
    videos = {
        "training_lame": [],
        "training_normal": [],
        "testing_ubc": []
    }

    # Training - Lame videos (label=1)
    if LAME_DIR.exists():
        all_lame = []
        for i in range(1, 26):
            path = LAME_DIR / f"L ({i}).mp4"
            if path.exists():
                all_lame.append((path, 1))
        # Apply skip and limit
        all_lame = all_lame[skip:]
        if limit > 0:
            all_lame = all_lame[:limit]
        videos["training_lame"] = all_lame

    # Training - Normal videos (label=0)
    if NORMAL_DIR.exists():
        all_normal = []
        for i in range(1, 26):
            path = NORMAL_DIR / f"N ({i}).mp4"
            if path.exists():
                all_normal.append((path, 0))
        # Apply skip and limit
        all_normal = all_normal[skip:]
        if limit > 0:
            all_normal = all_normal[:limit]
        videos["training_normal"] = all_normal

    # Testing - UBC videos (no label)
    if UBC_DIR.exists():
        all_ubc = [(path, None) for path in sorted(UBC_DIR.glob("*.mp4"))]
        # Apply skip and limit
        all_ubc = all_ubc[skip:]
        if limit > 0:
            all_ubc = all_ubc[:limit]
        videos["testing_ubc"] = all_ubc

    return videos


def print_discovery(videos: dict):
    """Print discovery summary."""
    print("\n" + "=" * 50)
    print("VIDEO DISCOVERY")
    print("=" * 50)
    print(f"Training - Lame (label=1):   {len(videos['training_lame'])} videos")
    print(f"Training - Normal (label=0): {len(videos['training_normal'])} videos")
    print(f"Testing - UBC (unlabeled):   {len(videos['testing_ubc'])} videos")
    total = sum(len(v) for v in videos.values())
    print(f"{'─' * 50}")
    print(f"Total: {total} videos")
    print("=" * 50 + "\n")


def print_progress(category: str, current: int, total: int, message: str = ""):
    """Print progress bar."""
    pct = (current / total) * 100 if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_len - filled)

    status = f"\r[{category:15}] [{bar}] {current}/{total} ({pct:5.1f}%)"
    if message:
        status += f" {message}"

    sys.stdout.write(status)
    sys.stdout.flush()


async def login(api_url: str, email: str, password: str) -> Optional[str]:
    """Login and get access token."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_url}/api/auth/login",
                json={"email": email, "password": password}
            )
            if response.status_code == 200:
                return response.json()["access_token"]
            else:
                print(f"Login failed: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"Login error: {e}")
        return None


async def upload_single(
    client: httpx.AsyncClient,
    api_url: str,
    file_path: Path,
    label: Optional[int]
) -> UploadResult:
    """Upload a single video with retry logic."""
    error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "video/mp4")}
                data = {}
                if label is not None:
                    data["label"] = str(label)

                response = await client.post(
                    f"{api_url}/api/videos/upload",
                    files=files,
                    data=data,
                    timeout=UPLOAD_TIMEOUT
                )

            if response.status_code == 200:
                result_data = response.json()
                return UploadResult(
                    file_path=str(file_path),
                    success=True,
                    video_id=result_data["video_id"],
                    label=label,
                    attempts=attempt
                )
            else:
                error = f"HTTP {response.status_code}: {response.text[:100]}"

        except httpx.TimeoutException:
            error = "Upload timed out"
        except httpx.ConnectError:
            error = "Connection failed"
        except Exception as e:
            error = str(e)[:100]

        # Exponential backoff before retry
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAY_BASE ** attempt
            await asyncio.sleep(delay)

    return UploadResult(
        file_path=str(file_path),
        success=False,
        label=label,
        error=error,
        attempts=MAX_RETRIES
    )


async def check_video_analyzed(
    client: httpx.AsyncClient,
    api_url: str,
    video_id: str
) -> bool:
    """Check if a video has been analyzed (has fusion results)."""
    try:
        response = await client.get(
            f"{api_url}/api/videos/{video_id}",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("has_analysis", False)
    except Exception:
        pass
    return False


async def wait_for_pipeline(
    client: httpx.AsyncClient,
    api_url: str,
    video_id: str,
    timeout: int = PIPELINE_TIMEOUT
) -> bool:
    """Wait for pipeline to finish processing a video."""
    start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start) < timeout:
        if await check_video_analyzed(client, api_url, video_id):
            return True
        await asyncio.sleep(PIPELINE_CHECK_INTERVAL)
    return False


async def upload_batch(
    api_url: str,
    files: List[Tuple[Path, Optional[int]]],
    category: str,
    delay: float = 0,
    wait_pipeline: bool = False
) -> BatchResult:
    """Upload a batch of videos with progress tracking.

    Args:
        api_url: Backend API URL
        files: List of (file_path, label) tuples
        category: Category name for progress display
        delay: Seconds to wait between uploads
        wait_pipeline: If True, wait for each video to be analyzed before next upload
    """
    result = BatchResult(category=category, total=len(files))

    async with httpx.AsyncClient() as client:
        for idx, (file_path, label) in enumerate(files):
            print_progress(category, idx + 1, len(files))

            upload_result = await upload_single(client, api_url, file_path, label)

            if upload_result.success:
                result.successful += 1
                result.video_ids.append(upload_result.video_id)

                # Wait for pipeline if requested
                if wait_pipeline and upload_result.video_id:
                    print_progress(category, idx + 1, len(files), "waiting for pipeline...")
                    pipeline_done = await wait_for_pipeline(
                        client, api_url, upload_result.video_id
                    )
                    if not pipeline_done:
                        print_progress(category, idx + 1, len(files), "timeout, continuing...")
            else:
                result.failed += 1
                result.errors.append(f"{file_path.name}: {upload_result.error}")

            # Add delay between uploads if specified
            if delay > 0 and idx < len(files) - 1:
                await asyncio.sleep(delay)

    print()  # New line after progress
    return result


async def trigger_batch_processing(
    api_url: str,
    token: str,
    video_ids: List[str],
    pipelines: Optional[List[str]] = None
) -> dict:
    """Trigger batch pipeline processing for uploaded videos."""
    headers = {"Authorization": f"Bearer {token}"}
    all_results = {"triggered": 0, "errors": 0}

    async with httpx.AsyncClient() as client:
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(video_ids), BATCH_SIZE):
            batch = video_ids[i:i + BATCH_SIZE]
            print(f"  Triggering batch {i//BATCH_SIZE + 1} ({len(batch)} videos)...")

            try:
                response = await client.post(
                    f"{api_url}/api/pipeline/batch/reprocess",
                    headers=headers,
                    json={
                        "video_ids": batch,
                        "pipelines": pipelines
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    all_results["triggered"] += data.get("triggered_count", 0)
                    all_results["errors"] += data.get("error_count", 0)
                else:
                    print(f"    Warning: Batch failed with {response.status_code}")
                    all_results["errors"] += len(batch)

            except Exception as e:
                print(f"    Warning: Batch error - {e}")
                all_results["errors"] += len(batch)

    return all_results


async def main(args):
    """Main upload orchestration."""
    start_time = datetime.now()

    # Discover videos
    videos = discover_videos(limit=args.limit, skip=args.skip)
    print_discovery(videos)

    if args.limit > 0 or args.skip > 0:
        info = []
        if args.skip > 0:
            info.append(f"skipping first {args.skip}")
        if args.limit > 0:
            info.append(f"limited to {args.limit}")
        print(f"({', '.join(info)} per category)\n")

    if args.dry_run:
        print("DRY RUN - No files will be uploaded\n")
        if videos["training_lame"]:
            print("Training Lame videos:")
            for path, _ in videos["training_lame"][:5]:
                print(f"  {path.name}")
            if len(videos["training_lame"]) > 5:
                print(f"  ... and {len(videos['training_lame']) - 5} more")

        if videos["training_normal"]:
            print("\nTraining Normal videos:")
            for path, _ in videos["training_normal"][:5]:
                print(f"  {path.name}")
            if len(videos["training_normal"]) > 5:
                print(f"  ... and {len(videos['training_normal']) - 5} more")

        if videos["testing_ubc"]:
            print("\nTesting UBC videos:")
            for path, _ in videos["testing_ubc"][:5]:
                print(f"  {path.name}")
            if len(videos["testing_ubc"]) > 5:
                print(f"  ... and {len(videos['testing_ubc']) - 5} more")
        return

    # Authenticate
    print("Authenticating...", end=" ")
    token = await login(args.api_url, args.email, args.password)
    if not token:
        print("FAILED")
        print("Error: Could not authenticate. Check credentials and ensure backend is running.")
        return
    print("OK\n")

    all_video_ids = []
    results = []

    # Upload training data
    if not args.skip_training:
        print("=" * 50)
        print("UPLOADING TRAINING DATA")
        print("=" * 50)

        if videos["training_lame"]:
            lame_result = await upload_batch(
                args.api_url,
                videos["training_lame"],
                "Training-Lame",
                delay=args.delay,
                wait_pipeline=args.wait_pipeline
            )
            results.append(lame_result)
            all_video_ids.extend(lame_result.video_ids)
            print(f"  Completed: {lame_result.successful}/{lame_result.total} successful")
            if lame_result.errors:
                for err in lame_result.errors[:3]:
                    print(f"    Error: {err}")

        if videos["training_normal"]:
            normal_result = await upload_batch(
                args.api_url,
                videos["training_normal"],
                "Training-Normal",
                delay=args.delay,
                wait_pipeline=args.wait_pipeline
            )
            results.append(normal_result)
            all_video_ids.extend(normal_result.video_ids)
            print(f"  Completed: {normal_result.successful}/{normal_result.total} successful")
            if normal_result.errors:
                for err in normal_result.errors[:3]:
                    print(f"    Error: {err}")
        print()

    # Upload testing data
    if not args.skip_testing:
        print("=" * 50)
        print("UPLOADING TESTING DATA")
        print("=" * 50)

        if videos["testing_ubc"]:
            ubc_result = await upload_batch(
                args.api_url,
                videos["testing_ubc"],
                "Testing-UBC",
                delay=args.delay,
                wait_pipeline=args.wait_pipeline
            )
            results.append(ubc_result)
            all_video_ids.extend(ubc_result.video_ids)
            print(f"  Completed: {ubc_result.successful}/{ubc_result.total} successful")
            if ubc_result.errors:
                for err in ubc_result.errors[:3]:
                    print(f"    Error: {err}")
                if len(ubc_result.errors) > 3:
                    print(f"    ... and {len(ubc_result.errors) - 3} more errors")
        print()

    # Trigger pipeline processing
    if not args.no_pipelines and all_video_ids:
        print("=" * 50)
        print("TRIGGERING PIPELINES")
        print("=" * 50)

        pipelines = args.pipelines.split(",") if args.pipelines else None
        print(f"Processing {len(all_video_ids)} videos...")

        pipeline_result = await trigger_batch_processing(
            args.api_url,
            token,
            all_video_ids,
            pipelines
        )
        print(f"  Triggered: {pipeline_result['triggered']}")
        print(f"  Errors: {pipeline_result['errors']}")
        print()

    # Print summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_uploaded = 0
    total_expected = 0

    for r in results:
        print(f"{r.category}: {r.successful}/{r.total} uploaded")
        total_uploaded += r.successful
        total_expected += r.total

    print(f"{'─' * 50}")
    print(f"Total: {total_uploaded}/{total_expected} videos uploaded")
    print(f"Video IDs collected: {len(all_video_ids)}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print("=" * 50)

    if all_video_ids:
        print(f"\nTo check results later, run:")
        print(f"  python scripts/run_pipeline_example.py results <video_id>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk upload videos for cow lameness detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bulk_upload.py                    # Full upload
  python scripts/bulk_upload.py --dry-run          # Preview only
  python scripts/bulk_upload.py --skip-testing     # Training only
  python scripts/bulk_upload.py --no-pipelines     # Upload without processing
        """
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Backend API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
        help="Admin email for authentication"
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help="Admin password"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without uploading"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training data upload"
    )
    parser.add_argument(
        "--skip-testing",
        action="store_true",
        help="Skip testing data upload"
    )
    parser.add_argument(
        "--no-pipelines",
        action="store_true",
        help="Skip triggering pipeline processing"
    )
    parser.add_argument(
        "--pipelines",
        help="Comma-separated list of pipelines to trigger (default: all)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=DEFAULT_LIMIT,
        help="Limit videos per category (0 = no limit, default: 0)"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N videos in each category (for resuming)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay in seconds between uploads (default: 0)"
    )
    parser.add_argument(
        "--wait-pipeline", "-w",
        action="store_true",
        help="Wait for pipeline to finish processing each video before uploading next"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
