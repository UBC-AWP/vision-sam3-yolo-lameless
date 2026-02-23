#!/usr/bin/env python3
"""
Check S3/Arbutus structure to see what paths exist.
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
    
    print(f"Checking bucket: {BUCKET}\n")
    
    # Check different prefixes
    prefixes_to_check = [
        "Farm1/",
        "Farm2/",
        "default/",
        "tenants/",
        "raw_videos/",
        "short_videos/",
        "clips_videos/",
    ]
    
    for prefix in prefixes_to_check:
        print(f"\n=== Checking prefix: {prefix} ===")
        paginator = s3.get_paginator("list_objects_v2")
        count = 0
        sample_keys = []
        
        try:
            for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, MaxKeys=10):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    if not key or key.endswith("/"):
                        continue
                    count += 1
                    if len(sample_keys) < 5:
                        sample_keys.append(key)
            
            print(f"Found {count} objects")
            if sample_keys:
                print("Sample keys:")
                for key in sample_keys[:5]:
                    print(f"  - {key}")
        except ClientError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
