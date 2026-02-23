#!/usr/bin/env python3
"""
Check if any raw videos are currently being processed.

Uses backend API:
  - GET /api/videos/storage/raw-videos  → each item has status_info.status:
      "queued" | "processing" | "completed" | "not_started" | "error"
  - GET /api/videos?status=processing    → videos with DB status "processing"

Usage:
    python scripts/check_processing_status.py
    API_URL=http://localhost:8000 python scripts/check_processing_status.py
    API_URL=... EMAIL=... PASSWORD=... python scripts/check_processing_status.py
"""

import os
import sys
import httpx

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_EMAIL = os.getenv("EMAIL", "admin@example.com")
DEFAULT_PASSWORD = os.getenv("PASSWORD", "adminpass123")


def login(base_url: str, email: str, password: str) -> str | None:
    r = httpx.post(
        f"{base_url}/api/auth/login",
        json={"email": email, "password": password},
        timeout=10,
    )
    if r.status_code != 200:
        print(f"Login failed: {r.status_code} {r.text}", file=sys.stderr)
        return None
    return r.json().get("access_token")


def main() -> None:
    base = DEFAULT_API_URL.rstrip("/")
    token = login(base, DEFAULT_EMAIL, DEFAULT_PASSWORD)
    if not token:
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}
    in_progress = []

    # 1) Raw videos (S3/storage) status from file-based status (queued.json / progress.json / long_video.json)
    try:
        r = httpx.get(f"{base}/api/videos/storage/raw-videos", headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("items") or []:
                info = item.get("status_info") or {}
                st = (info.get("status") or "").lower()
                if st in ("queued", "processing"):
                    in_progress.append({
                        "video_id": (item.get("filename") or "").replace(".mp4", "").replace(".avi", "").replace(".mov", "").replace(".mkv", "").strip("/"),
                        "source": "raw_videos",
                        "status": st,
                        "detail": info.get("detail") or info.get("reason") or "",
                        "percent": info.get("percent"),
                    })
        else:
            print(f"Raw videos list failed: {r.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"Error listing raw videos: {e}", file=sys.stderr)

    # 2) Videos in DB with status "processing"
    seen_ids = {x["video_id"] for x in in_progress}
    try:
        r = httpx.get(f"{base}/api/videos", params={"status": "processing", "limit": 100}, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            for v in data.get("videos") or []:
                vid = v.get("video_id") or v.get("id") or ""
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    in_progress.append({
                        "video_id": vid,
                        "source": "database",
                        "status": "processing",
                        "detail": "",
                        "percent": None,
                    })
        else:
            print(f"Videos list failed: {r.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"Error listing videos: {e}", file=sys.stderr)

    # Output
    if not in_progress:
        print("No raw videos are currently being processed.")
        return

    print(f"Found {len(in_progress)} raw video(s) in progress:\n")
    for x in in_progress:
        percent = f" {x['percent']}%" if x.get("percent") is not None else ""
        detail = f" — {x.get('detail') or ''}" if x.get("detail") else ""
        print(f"  - {x['video_id']}  [{x['source']}] {x['status']}{percent}{detail}")
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
