#!/usr/bin/env python3
"""
One-time migration: rename S3 objects from tenants/{tenant_id}/ to {tenant_name}/.

Old path: tenants/{tenant_id}/raw/{raw_video_id}/original.mp4
New path: {tenant_name}/raw/{raw_video_id}/original.mp4

Old path: tenants/{tenant_id}/clips/{raw_video_id}/{clip_slug}.mp4
New path: {tenant_name}/clips/{raw_video_id}/{clip_slug}.mp4

Requires: POSTGRES_URL or DATABASE_URL, S3_VIDEOS_BUCKET, and S3 credentials.
"""
import os
import sys
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

OLD_PREFIX = "tenants/"


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


def copy_s3_object(s3, source_key: str, dest_key: str) -> bool:
    """Copy S3 object from source to destination."""
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


def delete_s3_object(s3, key: str) -> bool:
    """Delete S3 object."""
    try:
        s3.delete_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        print(f"  Delete failed {key}: {e}")
        return False


def main():
    s3 = get_s3()
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print("Fetching tenant ID to name mapping from database...")
    cur.execute("SELECT id, name FROM tenants")
    tenant_map = {str(row["id"]): row["name"] for row in cur.fetchall()}
    print(f"Found {len(tenant_map)} tenants: {tenant_map}")

    print("\nScanning S3 for objects in tenants/...")
    
    # Find all objects in tenants/{tenant_id}/ paths
    paginator = s3.get_paginator("list_objects_v2")
    objects_to_migrate = []
    
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OLD_PREFIX):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            objects_to_migrate.append(key)
    
    if not objects_to_migrate:
        print("No objects found in tenants/ paths. Nothing to migrate.")
        conn.close()
        return
    
    print(f"Found {len(objects_to_migrate)} objects to migrate")
    
    migrated_count = 0
    failed_count = 0
    updated_db_count = 0
    
    for old_key in objects_to_migrate:
        # Parse tenant_id from path: tenants/{tenant_id}/raw/... or tenants/{tenant_id}/clips/...
        parts = old_key.split("/")
        if len(parts) < 3:
            print(f"  Skip {old_key}: invalid path format")
            continue
        
        tenant_id_str = parts[1]
        tenant_name = tenant_map.get(tenant_id_str)
        
        if not tenant_name:
            print(f"  Skip {old_key}: tenant_id {tenant_id_str} not found in database")
            continue
        
        # Build new key: {tenant_name}/raw/... or {tenant_name}/clips/...
        new_key = f"{tenant_name}/{'/'.join(parts[2:])}"
        
        # Check if new key already exists
        try:
            s3.head_object(Bucket=BUCKET, Key=new_key)
            print(f"  Skip {old_key}: destination already exists ({new_key})")
            # Still update DB if needed
            cur.execute(
                "UPDATE videos SET s3_key = %s WHERE s3_key = %s",
                (new_key, old_key)
            )
            if cur.rowcount > 0:
                updated_db_count += 1
            cur.execute(
                "UPDATE clip_videos SET s3_key = %s WHERE s3_key = %s",
                (new_key, old_key)
            )
            if cur.rowcount > 0:
                updated_db_count += 1
            continue
        except ClientError:
            pass  # Good, destination doesn't exist
        
        # Copy object to new location
        if copy_s3_object(s3, old_key, new_key):
            migrated_count += 1
            print(f"  Migrated: {old_key} -> {new_key}")
            
            # Update database
            cur.execute(
                "UPDATE videos SET s3_key = %s WHERE s3_key = %s",
                (new_key, old_key)
            )
            if cur.rowcount > 0:
                updated_db_count += 1
            
            cur.execute(
                "UPDATE clip_videos SET s3_key = %s WHERE s3_key = %s",
                (new_key, old_key)
            )
            if cur.rowcount > 0:
                updated_db_count += 1
            
            # Delete old object (optional - comment out if you want to keep both)
            if delete_s3_object(s3, old_key):
                print(f"    Deleted old object: {old_key}")
        else:
            failed_count += 1
    
    conn.commit()
    print(f"\nMigration complete:")
    print(f"  Objects migrated: {migrated_count}")
    print(f"  Database records updated: {updated_db_count}")
    print(f"  Failed: {failed_count}")
    
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
