#!/usr/bin/env python3
"""
Duplicate an existing Farm1 clip_1.mp4 into clip_2.mp4 and clip_3.mp4
for demo purposes, and register the new clips in the clip_videos table.
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
import boto3
import psycopg2


def main() -> None:
  load_dotenv()

  s3_endpoint = os.getenv("S3_ENDPOINT_URL")
  bucket = os.getenv("S3_VIDEOS_BUCKET")
  access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY_ID")
  secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_ACCESS_KEY")
  postgres_url = os.getenv(
    "POSTGRES_URL",
    "postgresql://lameness_user:lameness_pass@localhost:5432/lameness_db",
  )

  if not bucket:
    print("S3_VIDEOS_BUCKET is not set; aborting.")
    return

  s3_kwargs = {}
  if s3_endpoint:
    s3_kwargs["endpoint_url"] = s3_endpoint
  if access_key and secret_key:
    s3_kwargs["aws_access_key_id"] = access_key
    s3_kwargs["aws_secret_access_key"] = secret_key

  s3 = boto3.client("s3", **s3_kwargs)

  # 1. Find the first Farm1/clips/.../clip_1.mp4 as the source
  prefix = "Farm1/clips/"
  print(f"Searching for source clip under prefix: {prefix}")
  resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
  contents = resp.get("Contents") or []
  source_key = None
  for obj in contents:
    key = obj.get("Key") or ""
    if key.endswith("clip_1.mp4"):
      source_key = key
      break

  if not source_key:
    print("No Farm1/clips/.../clip_1.mp4 found; nothing to duplicate.")
    return

  print(f"Using source clip: {source_key}")
  dir_prefix = source_key.rsplit("/", 1)[0]
  clip2_key = f"{dir_prefix}/clip_2.mp4"
  clip3_key = f"{dir_prefix}/clip_3.mp4"

  # 2. Copy in S3
  for dest_key in (clip2_key, clip3_key):
    if dest_key == source_key:
      continue
    print(f"Copying {source_key} -> {dest_key}")
    s3.copy_object(
      Bucket=bucket,
      CopySource={"Bucket": bucket, "Key": source_key},
      Key=dest_key,
    )

  # 3. Duplicate DB rows in clip_videos
  print("Connecting to PostgreSQL to duplicate clip_videos rows...")
  conn = psycopg2.connect(postgres_url)
  cur = conn.cursor()

  cur.execute(
    """
    SELECT id, tenant_id, raw_video_id, clip_slug, s3_key,
           start_sec, end_sec, quality_label, label, prob_bad
    FROM clip_videos
    WHERE s3_key = %s
    """,
    (source_key,),
  )
  row = cur.fetchone()
  if not row:
    print("No clip_videos row found for source clip; DB not modified.")
    conn.close()
    return

  (
    _,
    tenant_id,
    raw_video_id,
    clip_slug,
    s3_key,
    start_sec,
    end_sec,
    quality_label,
    label,
    prob_bad,
  ) = row

  for new_slug, new_key in (("clip_2", clip2_key), ("clip_3", clip3_key)):
    new_id = uuid.uuid4()
    print(f"Inserting DB row for {new_slug} -> {new_key}")
    cur.execute(
      """
      INSERT INTO clip_videos (
        id, tenant_id, raw_video_id, clip_slug, s3_key,
        start_sec, end_sec, quality_label, label, prob_bad, created_at
      ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, NOW()
      )
      """,
      (
        str(new_id),
        tenant_id,
        raw_video_id,
        new_slug,
        new_key,
        start_sec,
        end_sec,
        quality_label,
        label,
        prob_bad,
      ),
    )

  conn.commit()
  conn.close()
  print("Demo clips clip_2 and clip_3 created and registered.")


if __name__ == "__main__":
  main()

