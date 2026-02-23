#!/usr/bin/env python3
"""
Create Farm1 and Farm2 folder structure in S3/Arbutus by uploading placeholder files.
This makes the folders visible in the Arbutus UI.
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
    import httpx
    import asyncio
except ImportError:
    print("Install boto3 and httpx: pip install boto3 httpx")
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

async def create_placeholders():
    s3 = get_s3()
    
    # Create placeholder files to establish folder structure
    farms = ["Farm1", "Farm2"]
    folders = ["raw", "clips"]
    
    print("Creating folder structure in S3/Arbutus...")
    
    for farm in farms:
        for folder in folders:
            # Create a placeholder file
            placeholder_key = f"{farm}/{folder}/.placeholder"
            placeholder_content = b"This is a placeholder file to create the folder structure."
            
            try:
                # Check if placeholder already exists
                s3.head_object(Bucket=BUCKET, Key=placeholder_key)
                print(f"  {placeholder_key} already exists, skipping")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # For Arbutus, use presigned URL with explicit Content-Length
                    try:
                        presigned_url = s3.generate_presigned_url(
                            "put_object",
                            Params={
                                "Bucket": BUCKET,
                                "Key": placeholder_key,
                                "ContentType": "text/plain",
                            },
                            ExpiresIn=3600,
                        )
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            r = await client.put(
                                presigned_url,
                                content=placeholder_content,
                                headers={
                                    "Content-Type": "text/plain",
                                    "Content-Length": str(len(placeholder_content)),
                                },
                            )
                        if r.status_code >= 400:
                            print(f"  Failed to create {placeholder_key}: {r.status_code} {r.text[:200]}")
                        else:
                            print(f"  Created: {placeholder_key}")
                    except Exception as upload_error:
                        # Fallback: try direct put_object
                        try:
                            s3.put_object(
                                Bucket=BUCKET,
                                Key=placeholder_key,
                                Body=placeholder_content,
                                ContentType="text/plain",
                            )
                            print(f"  Created (fallback): {placeholder_key}")
                        except Exception as fallback_error:
                            print(f"  Error creating {placeholder_key}: {fallback_error}")
                else:
                    print(f"  Error checking {placeholder_key}: {e}")
    
    print("\nDone! Farm1 and Farm2 folders should now be visible in Arbutus.")
    print("You can delete the .placeholder files later if needed.")

def main():
    asyncio.run(create_placeholders())

if __name__ == "__main__":
    main()
