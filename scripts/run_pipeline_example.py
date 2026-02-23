#!/usr/bin/env python3
"""
Example script demonstrating how to:
1. Upload a video and trigger the full pipeline
2. Monitor pipeline progress via NATS
3. Read results from each pipeline

Usage:
    python scripts/run_pipeline_example.py <video_path>
    python scripts/run_pipeline_example.py --list-results <video_id>
"""

import asyncio
import json
import sys
import httpx
from pathlib import Path

# Service URLs (when running locally with docker-compose)
VIDEO_INGESTION_URL = "http://localhost:8001"
ADMIN_BACKEND_URL = "http://localhost:8000"


async def upload_video(video_path: str):
    """Upload a video and trigger the full processing pipeline."""
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return None

    print(f"Uploading video: {video_path}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(video_path, "rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            response = await client.post(
                f"{VIDEO_INGESTION_URL}/upload",
                files=files
            )

        if response.status_code == 200:
            result = response.json()
            print(f"Video uploaded successfully!")
            print(f"  Video ID: {result['video_id']}")
            print(f"  File size: {result['file_size']} bytes")
            print(f"  Uploaded at: {result['uploaded_at']}")
            print()
            print("Pipeline processing has been triggered automatically.")
            print("NATS flow: video.uploaded -> video.preprocessed -> all pipelines")
            return result['video_id']
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")
            return None


async def get_pipeline_results(video_id: str):
    """Fetch results from all pipelines for a specific video."""

    print(f"\n=== Pipeline Results for {video_id} ===\n")

    # Results are stored in /app/data/results/<pipeline>/<video_id>_<pipeline>.json
    # We can access them via the admin backend API or directly from the container

    pipelines = ['yolo', 'sam3', 'dinov3', 'tleap', 'tcn', 'transformer', 'gnn', 'ml', 'fusion']

    for pipeline in pipelines:
        print(f"\n--- {pipeline.upper()} Pipeline ---")

        # Try to read from local data directory
        result_path = Path(f"data/results/{pipeline}/{video_id}_{pipeline}.json")

        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)

            # Show key metrics based on pipeline type
            if pipeline == 'yolo':
                print(f"  Detections: {result.get('features', {}).get('num_detections', 'N/A')}")
                print(f"  Avg Confidence: {result.get('features', {}).get('avg_confidence', 'N/A'):.3f}")

            elif pipeline == 'tleap':
                features = result.get('locomotion_features', {})
                print(f"  Frames Processed: {result.get('frames_processed', 'N/A')}")
                print(f"  Lameness Score: {features.get('lameness_score', 'N/A')}")
                print(f"  Head Bob Magnitude: {features.get('head_bob_magnitude', 'N/A')}")

            elif pipeline == 'tcn':
                print(f"  Severity Score: {result.get('severity_score', 'N/A'):.3f}")
                print(f"  Uncertainty: {result.get('uncertainty', 'N/A'):.4f}")
                print(f"  Prediction: {'Lame' if result.get('prediction') == 1 else 'Healthy'}")

            elif pipeline == 'transformer':
                print(f"  Severity Score: {result.get('severity_score', 'N/A')}")
                print(f"  Uncertainty: {result.get('uncertainty', 'N/A')}")

            elif pipeline == 'gnn':
                print(f"  Severity Score: {result.get('severity_score', 'N/A')}")
                print(f"  Neighbor Influence: {result.get('neighbor_influence', [])}")

            elif pipeline == 'dinov3':
                print(f"  Neighbor Evidence: {result.get('neighbor_evidence', 'N/A')}")
                print(f"  Similar Cases: {len(result.get('similar_cases', []))}")

            elif pipeline == 'ml':
                ensemble = result.get('ensemble', {})
                print(f"  Ensemble Probability: {ensemble.get('probability', 'N/A')}")
                print(f"  Prediction: {'Lame' if ensemble.get('prediction') == 1 else 'Healthy'}")

            elif pipeline == 'fusion':
                fusion = result.get('fusion_result', {})
                final_prob = fusion.get('final_probability')
                if final_prob is not None:
                    print(f"  FINAL Score: {final_prob:.3f}")
                else:
                    print(f"  FINAL Score: N/A")
                print(f"  FINAL Prediction: {'LAME' if fusion.get('final_prediction') == 1 else 'HEALTHY'}")
                print(f"  Pipeline Contributions:")
                for name, value in fusion.get('pipeline_contributions', {}).items():
                    if value is not None and isinstance(value, (int, float)):
                        print(f"    - {name}: {value:.3f}")
                    elif value is not None:
                        print(f"    - {name}: {value}")
        else:
            print(f"  [Result not available yet]")


async def trigger_single_pipeline(video_id: str, pipeline: str):
    """
    Manually trigger a single pipeline via the admin API.
    Requires admin authentication.
    """
    async with httpx.AsyncClient() as client:
        # First, login to get token
        login_response = await client.post(
            f"{ADMIN_BACKEND_URL}/api/auth/login",
            json={"email": "admin@example.com", "password": "adminpass123"}
        )

        if login_response.status_code != 200:
            print("Login failed. Make sure admin user exists.")
            return

        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Trigger the pipeline
        response = await client.post(
            f"{ADMIN_BACKEND_URL}/api/pipeline/{pipeline}/trigger/{video_id}",
            headers=headers
        )

        if response.status_code == 200:
            print(f"Successfully triggered {pipeline} pipeline for {video_id}")
            return response.json()
        else:
            print(f"Failed to trigger: {response.status_code} - {response.text}")
            return None


async def check_pipeline_status():
    """Check the status of all pipeline services."""
    async with httpx.AsyncClient() as client:
        # Login
        login_response = await client.post(
            f"{ADMIN_BACKEND_URL}/api/auth/login",
            json={"email": "admin@example.com", "password": "adminpass123"}
        )

        if login_response.status_code != 200:
            print("Login failed")
            return

        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Get pipeline status
        response = await client.get(
            f"{ADMIN_BACKEND_URL}/api/pipeline/status",
            headers=headers
        )

        if response.status_code == 200:
            pipelines = response.json()
            print("\n=== Pipeline Service Status ===\n")
            for p in pipelines:
                status = p.get('status', 'unknown')
                icon = "✓" if status == 'healthy' else "○" if status == 'unknown' else "✗"
                print(f"  {icon} {p['service_name']}: {status}")
                if p.get('active_jobs'):
                    print(f"    Active jobs: {p['active_jobs']}")


def print_usage():
    print("""
Pipeline Runner - Example Script
================================

Usage:
    # Upload a video and trigger full pipeline
    python scripts/run_pipeline_example.py upload <video_path>

    # Get results for a processed video
    python scripts/run_pipeline_example.py results <video_id>

    # Check pipeline service status
    python scripts/run_pipeline_example.py status

    # Trigger a single pipeline manually
    python scripts/run_pipeline_example.py trigger <video_id> <pipeline_name>

Pipeline names: yolo, sam3, dinov3, tleap, tcn, transformer, gnn, ml

Example:
    python scripts/run_pipeline_example.py upload data/videos/cow_walking.mp4
    python scripts/run_pipeline_example.py results 05e23393-3ae5-4287-8a3f-ad966be3e28c
    python scripts/run_pipeline_example.py status
    python scripts/run_pipeline_example.py trigger 05e23393-3ae5-4287-8a3f-ad966be3e28c yolo
""")


async def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "upload" and len(sys.argv) >= 3:
        video_path = sys.argv[2]
        video_id = await upload_video(video_path)
        if video_id:
            print("\nWait a few seconds for pipelines to process, then run:")
            print(f"  python scripts/run_pipeline_example.py results {video_id}")

    elif command == "results" and len(sys.argv) >= 3:
        video_id = sys.argv[2]
        await get_pipeline_results(video_id)

    elif command == "status":
        await check_pipeline_status()

    elif command == "trigger" and len(sys.argv) >= 4:
        video_id = sys.argv[2]
        pipeline = sys.argv[3]
        await trigger_single_pipeline(video_id, pipeline)

    else:
        print_usage()


if __name__ == "__main__":
    asyncio.run(main())
