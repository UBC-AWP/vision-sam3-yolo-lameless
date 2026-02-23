#!/usr/bin/env python3
"""
One-time migration: copy S3 objects from legacy keys to tenant-prefixed keys and update DB.

Legacy:
  raw_videos/{tenant_name}/{filename}.mp4
  short_videos/{raw_video_id}/{filename}.mp4

New:
  tenants/{tenant_id}/raw/{raw_video_id}/original.mp4
  tenants/{tenant_id}/short/{raw_video_id}/clip_N.mp4

Requires: POSTGRES_URL or DATABASE_URL, S3_VIDEOS_BUCKET, and S3 credentials.
Run SQL migration first: psql -f scripts/migrate_tenant_video_storage.sql
"""
import os
import sys
import uuid
from pathlib import Path

# Load .env if present
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
except ImportError:
    print("Install boto3: pip install boto3")
    sys.exit(1)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Install psycopg2-binary: pip install psycopg2-binary")
    sys.exit(1)

BUCKET = os.getenv("S3_VIDEOS_BUCKET")
if not BUCKET:
    print("Set S3_VIDEOS_BUCKET")
    sys.exit(1)

_raw_url = os.getenv("POSTGRES_URL", os.getenv("DATABASE_URL", ""))
if not _raw_url:
    print("Set POSTGRES_URL or DATABASE_URL")
    sys.exit(1)

# Sync driver for script
DATABASE_URL = _raw_url.replace("postgresql+asyncpg://", "postgresql://").replace("asyncpg://", "postgresql://")

TENANTS_PREFIX = "tenants/"
RAW_LEGACY = "raw_videos/"
SHORT_LEGACY = "short_videos/"


def get_s3():
    kwargs = {"region_name": os.getenv("AWS_REGION", "us-west-2")}
    if os.getenv("S3_ENDPOINT_URL"):
        kwargs["endpoint_url"] = os.getenv("S3_ENDPOINT_URL")
    if os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID"):
        kwargs["aws_access_key_id"] = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
        kwargs["aws_secret_access_key"] = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    if os.getenv("S3_FORCE_PATH_STYLE", "").lower() in ("1", "true", "yes"):
        kwargs["config"] = Config(s3={"addressing_style": "path"})
    return boto3.client("s3", **kwargs)


def copy_s3(s3, source_key: str, dest_key: str) -> bool:
    try:
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": source_key},
            Key=dest_key,
        )
        return True
    except ClientError as e:
        print(f"  Copy failed {source_key} -> {dest_key}: {e}")
        return False


def main():
    s3 = get_s3()
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False

    # 1) Migrate raw videos: Video rows with s3_key under raw_videos/
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT id, s3_key, tenant_id FROM videos WHERE s3_key IS NOT NULL AND s3_key LIKE %s",
        (f"{RAW_LEGACY}%",),
    )
    raw_rows = cur.fetchall()
    raw_migrated = 0
    for row in raw_rows:
        vid, key, tid = row["id"], row["s3_key"], row["tenant_id"]
        if not tid:
            parts = (key or "").split("/")
            if len(parts) >= 2 and parts[0].rstrip("/") == RAW_LEGACY.rstrip("/"):
                tenant_name = parts[1]
                cur.execute("SELECT id FROM tenants WHERE name = %s", (tenant_name,))
                trow = cur.fetchone()
                if trow:
                    tid = trow["id"]
                    cur.execute("UPDATE videos SET tenant_id = %s WHERE id = %s", (tid, vid))
            if not tid:
                print(f"  Skip video {vid}: no tenant_id")
                continue
        tid = str(tid)
        new_key = f"{TENANTS_PREFIX}{tid}/raw/{vid}/original.mp4"
        if key == new_key:
            continue
        if copy_s3(s3, key, new_key):
            cur.execute("UPDATE videos SET s3_key = %s WHERE id = %s", (new_key, vid))
            raw_migrated += 1
            print(f"  Raw: {vid} -> {new_key}")
    conn.commit()
    print(f"Raw videos migrated: {raw_migrated}")

    # 2) List short_videos prefix and group by raw_video_id
    paginator = s3.get_paginator("list_objects_v2")
    short_objects = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=SHORT_LEGACY):
        for obj in page.get("Contents", []):
            k = obj.get("Key", "")
            if not k or k.endswith("/"):
                continue
            parts = k.split("/")
            if len(parts) >= 3:
                raw_id = parts[1]
                short_objects.append((k, raw_id, parts[2]))
    if not short_objects:
        print("No legacy short videos to migrate.")
        conn.close()
        return

    # Get tenant_id per raw_video_id
    raw_ids = list({r[1] for r in short_objects})
    cur.execute(
        "SELECT id, tenant_id FROM videos WHERE id = ANY(%s)",
        (raw_ids,),
    )
    raw_to_tenant = {str(row["id"]): str(row["tenant_id"]) for row in cur.fetchall() if row["tenant_id"]}

    clip_index_per_raw = {}
    clip_migrated = 0
    clips_to_insert = []
    for old_key, raw_id, filename in short_objects:
        tid = raw_to_tenant.get(raw_id)
        if not tid:
            print(f"  Skip short {old_key}: no tenant for raw {raw_id}")
            continue
        clip_index_per_raw[raw_id] = clip_index_per_raw.get(raw_id, 0) + 1
        clip_slug = f"clip_{clip_index_per_raw[raw_id]}"
        new_key = f"{TENANTS_PREFIX}{tid}/short/{raw_id}/{clip_slug}.mp4"
        if copy_s3(s3, old_key, new_key):
            clip_migrated += 1
            clips_to_insert.append((str(uuid.uuid4()), tid, raw_id, clip_slug, new_key))
            print(f"  Short: {old_key} -> {new_key}")

    for cid, tid, raw_id, slug, sk in clips_to_insert:
        cur.execute(
            """INSERT INTO clip_videos (id, tenant_id, raw_video_id, clip_slug, s3_key)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (s3_key) DO NOTHING""",
            (cid, tid, raw_id, slug, sk),
        )
    conn.commit()
    print(f"Short clips migrated: {clip_migrated}, clip_videos inserted: {len(clips_to_insert)}")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
