#!/usr/bin/env python3
"""
List all S3 keys to see the actual structure.
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

BUCKET = os.getenv("S3_VIDEOS_BUCKET")
if not BUCKET:
    print("Set S3_VIDEOS_BUCKET")
    sys.exit(1)

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

def main():
    s3 = get_s3()
    
    print(f"Listing all objects in bucket: {BUCKET}\n")
    
    paginator = s3.get_paginator("list_objects_v2")
    all_keys = []
    
    try:
        for page in paginator.paginate(Bucket=BUCKET):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key and not key.endswith("/"):
                    all_keys.append(key)
    except ClientError as e:
        print(f"Error listing objects: {e}")
        return
    
    # Group by top-level prefix
    prefixes = {}
    for key in sorted(all_keys):
        top_level = key.split("/")[0] if "/" in key else key
        if top_level not in prefixes:
            prefixes[top_level] = []
        prefixes[top_level].append(key)
    
    print(f"Total objects: {len(all_keys)}\n")
    print("Structure by top-level prefix:")
    print("=" * 60)
    
    for prefix in sorted(prefixes.keys()):
        print(f"\n{prefix}/ ({len(prefixes[prefix])} objects)")
        for key in prefixes[prefix][:10]:  # Show first 10
            print(f"  - {key}")
        if len(prefixes[prefix]) > 10:
            print(f"  ... and {len(prefixes[prefix]) - 10} more")

if __name__ == "__main__":
    main()
