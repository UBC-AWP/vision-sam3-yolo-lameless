"""
Video management endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form, Response, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Set
from urllib.parse import urlparse
from pathlib import Path
import json
import uuid
from datetime import datetime, timedelta
import cv2
import io
import httpx
import asyncio
import os
import nats
import boto3
import yaml
import re
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..database import get_db, Video, Tenant, User, ClipVideo
from ..middleware.auth import get_current_user

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure debug logs are shown

router = APIRouter()

VIDEOS_DIR = Path("/app/data/videos")
PROCESSED_DIR = Path("/app/data/processed")
ANNOTATED_DIR = PROCESSED_DIR / "annotated"
RESULTS_DIR = Path("/app/data/results")
TRAINING_DIR = Path("/app/data/training")
LONG_VIDEO_RESULTS_DIR = RESULTS_DIR / "long_video"

# Storage backend configuration
# STORAGE_BACKEND: "local" (default for development) or "s3" (for AWS deployment)
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
S3_VIDEOS_BUCKET = os.getenv("S3_VIDEOS_BUCKET", "")
CLOUDFRONT_DOMAIN = os.getenv("CLOUDFRONT_DOMAIN", "")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_FORCE_PATH_STYLE = os.getenv("S3_FORCE_PATH_STYLE", "false").lower() in {"1", "true", "yes"}
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
DEFAULT_TENANT_NAME = os.getenv("DEFAULT_TENANT_NAME", "default")
# New tenant-prefixed key format: {tenant_name}/raw|clips/...
# Examples: Farm1/raw/{raw_video_id}/original.mp4, Farm2/clips/{raw_video_id}/{clip_slug}.mp4
RAW_VIDEOS_PREFIX = os.getenv("S3_LONG_VIDEOS_PREFIX", "raw_videos/")  # legacy; migration only
CLIPS_VIDEOS_PREFIX = os.getenv("S3_CLIPS_VIDEOS_PREFIX", "clips_videos/")  # legacy; migration only
SHORT_VIDEOS_PREFIX = os.getenv("S3_SHORT_VIDEOS_PREFIX", "short_videos/")  # legacy; for backward compatibility
TENANTS_PREFIX = "tenants/"  # legacy; for backward compatibility


async def _get_tenant_name(db: AsyncSession, tenant_id) -> str:
    """Get tenant name from tenant_id. Returns 'default' if not found."""
    if not tenant_id:
        return DEFAULT_TENANT_NAME
    result = await db.execute(select(Tenant.name).where(Tenant.id == tenant_id))
    row = result.one_or_none()
    if not row:
        return DEFAULT_TENANT_NAME
    return (row[0] or DEFAULT_TENANT_NAME).strip()


def _raw_s3_key(tenant_name: str, raw_video_id: str) -> str:
    """New key for raw video: {tenant_name}/raw/{raw_video_id}/original.mp4"""
    return f"{tenant_name}/raw/{raw_video_id}/original.mp4"


def _clips_s3_key(tenant_name: str, raw_video_id: str, clip_slug: str) -> str:
    """New key for clip video: {tenant_name}/clips/{raw_video_id}/{clip_slug}.mp4"""
    return f"{tenant_name}/clips/{raw_video_id}/{clip_slug}.mp4"

# Initialize S3 client (lazy loaded)
_s3_client = None

def get_s3_client():
    """Get S3 client (lazy initialization)"""
    global _s3_client
    if _s3_client is None and S3_VIDEOS_BUCKET:
        client_kwargs = {
            "region_name": AWS_REGION,
        }
        if S3_ENDPOINT_URL:
            client_kwargs["endpoint_url"] = S3_ENDPOINT_URL
        if S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY:
            client_kwargs["aws_access_key_id"] = S3_ACCESS_KEY_ID
            client_kwargs["aws_secret_access_key"] = S3_SECRET_ACCESS_KEY
        if S3_FORCE_PATH_STYLE:
            client_kwargs["config"] = Config(s3={"addressing_style": "path"})
        _s3_client = boto3.client("s3", **client_kwargs)
    return _s3_client

def is_s3_enabled():
    """Check if S3 storage is enabled.

    S3 is enabled when:
    1. STORAGE_BACKEND is set to "s3", AND
    2. S3_VIDEOS_BUCKET is configured

    This allows local development to use local storage even if S3 bucket is configured.
    """
    return STORAGE_BACKEND == "s3" and bool(S3_VIDEOS_BUCKET)

def get_storage_backend():
    """Get the current storage backend type."""
    if is_s3_enabled():
        return "s3"
    return "local"

# Annotation renderer service URL (uses service discovery namespace)
SERVICE_NAMESPACE = os.getenv("SERVICE_NAMESPACE", "cow-lameness-production.local")
ANNOTATION_RENDERER_URL = f"http://annotation-renderer.{SERVICE_NAMESPACE}:8000"

# NATS connection for triggering pipelines
_nats_client = None
_config_cache = None

async def get_nats():
    """Get NATS connection"""
    global _nats_client
    if _nats_client is None or not _nats_client.is_connected:
        nats_url = os.getenv("NATS_URL", "nats://nats:4222")
        _nats_client = await nats.connect(nats_url)
    return _nats_client


def _load_shared_config():
    global _config_cache
    if _config_cache is None:
        config_path = Path("/app/shared/config/config.yaml")
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f) or {}
        else:
            _config_cache = {}
    return _config_cache


def _get_subject(subject_key: str, default: str) -> str:
    config = _load_shared_config()
    return config.get("nats", {}).get("subjects", {}).get(subject_key, default)


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _list_s3_objects(prefix: str) -> List[Dict[str, Any]]:
    if not is_s3_enabled():
        raise HTTPException(status_code=400, detail="S3 storage not configured")
    s3 = get_s3_client()
    if s3 is None:
        raise HTTPException(status_code=500, detail="S3 client not initialized")

    prefix = _normalize_prefix(prefix)
    paginator = s3.get_paginator("list_objects_v2")
    items: List[Dict[str, Any]] = []
    for page in paginator.paginate(Bucket=S3_VIDEOS_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            suffix = Path(key).suffix.lower()
            # Also check uppercase extensions
            if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"} and Path(key).suffix.upper() not in {".MP4", ".AVI", ".MOV", ".MKV", ".M4V"}:
                continue
            items.append({
                "key": key,
                "filename": Path(key).name,
                "size": obj.get("Size", 0),
                "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
                "etag": obj.get("ETag", "").strip('"'),
            })
    return items


def _presigned_url_for_key(key: str) -> Optional[str]:
    if not is_s3_enabled():
        return None
    s3 = get_s3_client()
    if s3 is None:
        return None
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_VIDEOS_BUCKET, "Key": key},
            ExpiresIn=3600
        )
    except ClientError:
        return None


async def _s3_has_clips_for(db: AsyncSession, tenant_id: Optional[str], video_id: str) -> bool:
    """Return True if S3 has at least one clip video for this raw. Checks both new clips/ path and legacy paths."""
    if not video_id or not is_s3_enabled():
        return True
    s3 = get_s3_client()
    if s3 is None:
        return True
    prefixes_to_check = []
    if tenant_id:
        # Get tenant name
        tenant_name = await _get_tenant_name(db, tenant_id)
        # Check new path: {tenant_name}/clips/
        prefixes_to_check.append(_normalize_prefix(f"{tenant_name}/clips/") + video_id.strip() + "/")
        # Also check legacy paths for backward compatibility
        prefixes_to_check.append(_normalize_prefix(f"{TENANTS_PREFIX}{str(tenant_id)}/clips/") + video_id.strip() + "/")
        prefixes_to_check.append(_normalize_prefix(f"{TENANTS_PREFIX}{str(tenant_id)}/short/") + video_id.strip() + "/")
    else:
        prefixes_to_check.append(_normalize_prefix(CLIPS_VIDEOS_PREFIX) + video_id.strip() + "/")
        prefixes_to_check.append(_normalize_prefix(SHORT_VIDEOS_PREFIX) + video_id.strip() + "/")
    
    for prefix in prefixes_to_check:
        try:
            resp = s3.list_objects_v2(Bucket=S3_VIDEOS_BUCKET, Prefix=prefix, MaxKeys=1)
            contents = resp.get("Contents")
            if contents is None:
                contents = resp.get("contents") or []
            if len(contents) > 0:
                return True
        except ClientError:
            continue
    return False


def _load_long_video_status(video_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    progress_path = LONG_VIDEO_RESULTS_DIR / f"{video_id}_progress.json"
    progress_data = None
    if progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as f:
                progress_data = json.load(f) or {}
        except Exception:
            progress_data = None

    summary_path = LONG_VIDEO_RESULTS_DIR / f"{video_id}_long_video.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            status = data.get("status", "completed")
            reason = data.get("reason", "")
            upload_error = data.get("upload_error", "")
            upload_failed_count = data.get("upload_failed_count", 0)
            if status == "completed":
                detail = "Completed"
            elif status == "upload_failed":
                detail = upload_error or reason or "Clips not uploaded to S3"
            else:
                detail = reason or "Error"
            result = {
                "status": status,
                "reason": reason,
                "good_count": data.get("good_count", 0),
                "bad_count": data.get("bad_count", 0),
                "updated_at": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(),
                "percent": 100 if status in ("completed", "upload_failed") else (progress_data.get("percent", 0) if progress_data else 0),
                "detail": detail,
            }
            if upload_failed_count > 0:
                result["upload_failed_count"] = upload_failed_count
            if upload_error:
                result["upload_error"] = upload_error
            return result
        except Exception as e:
            return {
                "status": "error",
                "reason": f"Failed to read summary: {e}",
                "good_count": 0,
                "bad_count": 0,
                "updated_at": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(),
                "percent": 0,
                "detail": "Error",
            }

    queued_path = LONG_VIDEO_RESULTS_DIR / f"{video_id}_queued.json"
    if queued_path.exists():
        try:
            with queued_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            return {
                "status": data.get("status", "queued"),
                "reason": data.get("reason", ""),
                "good_count": 0,
                "bad_count": 0,
                "updated_at": data.get("queued_at"),
                "percent": progress_data.get("percent") if progress_data else 0,
                "detail": progress_data.get("detail") if progress_data else "Queued",
            }
        except Exception:
            return {
                "status": "queued",
                "reason": "",
                "good_count": 0,
                "bad_count": 0,
                "updated_at": datetime.fromtimestamp(queued_path.stat().st_mtime).isoformat(),
                "percent": progress_data.get("percent") if progress_data else 0,
                "detail": progress_data.get("detail") if progress_data else "Queued",
            }

    if progress_data:
        return {
            "status": progress_data.get("status", "processing"),
            "reason": "",
            "good_count": 0,
            "bad_count": 0,
            "updated_at": datetime.fromtimestamp(progress_path.stat().st_mtime).isoformat(),
            "percent": progress_data.get("percent", 0),
            "detail": progress_data.get("detail", ""),
        }

    return {
        "status": "not_started",
        "reason": "",
        "good_count": 0,
        "bad_count": 0,
        "updated_at": None,
        "percent": 0,
        "detail": "",
    }


def _build_s3_key_to_status() -> Dict[str, Dict[str, Any]]:
    """Build map from s3_key to status_info from summary files that have s3_key (so raw videos match even when video_id from list doesn't match)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not LONG_VIDEO_RESULTS_DIR.exists():
        return out
    for summary_path in LONG_VIDEO_RESULTS_DIR.glob("*_long_video.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            continue
        key = data.get("s3_key")
        if not key:
            continue
        status_val = data.get("status", "completed")
        video_id = data.get("video_id", "")
        # Parse tenant_id from new key format tenants/{tid}/raw/...
        tid = None
        if key and key.startswith(TENANTS_PREFIX):
            parts = key.split("/")
            if len(parts) >= 2:
                tid = parts[1]
        # Only show completed if S3 has clips for this raw video; if deleted, show as not started.
        # NOTE: The old implementation called _s3_has_clips_for(tid, video_id), but this helper
        # was refactored to be async and requires a DB session. Since _build_s3_key_to_status is
        # a synchronous helper used only to enrich raw video status, and the clips-existence check
        # is non-critical for listing raw videos, we temporarily disable this extra S3 lookup
        # to avoid runtime errors and let the summary file's status drive the UI.
        #
        # If you want to re‑enable this in the future, create a lightweight, synchronous helper
        # that checks S3 for clips based solely on the stored s3_key instead of (tenant_id, video_id).
        # For now we skip overriding the status here.
        # if status_val == "completed" and video_id and not _s3_has_clips_for(tid, video_id):
        #     out[key] = { ... }
        reason = data.get("reason", "")
        upload_error = data.get("upload_error", "")
        upload_failed_count = data.get("upload_failed_count", 0)
        if status_val == "completed":
            detail = "Completed"
        elif status_val == "upload_failed":
            detail = upload_error or reason or "Short clips not uploaded to S3"
        else:
            detail = reason or "Error"
        entry = {
            "status": status_val,
            "reason": reason,
            "good_count": data.get("good_count", 0),
            "bad_count": data.get("bad_count", 0),
            "updated_at": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat() if summary_path.exists() else None,
            "percent": 100 if status_val in ("completed", "upload_failed") else 0,
            "detail": detail,
        }
        if upload_failed_count > 0:
            entry["upload_failed_count"] = upload_failed_count
        if upload_error:
            entry["upload_error"] = upload_error
        out[key] = entry
    return out


def _build_clips_label_map() -> Dict[str, Dict[str, Any]]:
    label_map: Dict[str, Dict[str, Any]] = {}
    if not LONG_VIDEO_RESULTS_DIR.exists():
        return label_map
    for summary_path in LONG_VIDEO_RESULTS_DIR.glob("*_long_video.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            continue
        parent_id = data.get("video_id")
        for entry in data.get("good_short_videos", []):
            file_name = entry.get("file_name")
            if not file_name:
                continue
            label_map[file_name] = {
                "label": "good",
                "prob_bad": entry.get("prob_bad"),
                "parent_video_id": parent_id,
            }
        for entry in data.get("bad_short_videos", []):
            file_name = entry.get("file_name")
            if not file_name:
                continue
            label_map[file_name] = {
                "label": "bad",
                "prob_bad": entry.get("prob_bad"),
                "parent_video_id": parent_id,
            }
    return label_map


def _parent_display_name_from_summary(video_id: str) -> Optional[str]:
    """Resolve human-readable original filename from pipeline summary file (when Video row is missing or has no original_filename)."""
    path = LONG_VIDEO_RESULTS_DIR / f"{video_id}_long_video.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return None
    s3_key = data.get("s3_key")
    if s3_key:
        name = Path(s3_key).name
        if name:
            return name
    original_path = data.get("original_path")
    if not original_path:
        return None
    basename = Path(original_path).name
    if not basename:
        return None
    if basename.endswith("_compressed.mp4"):
        stem = basename[:-len("_compressed.mp4")]
        return stem + ".mp4"
    if basename.endswith("_compressed.MP4"):
        stem = basename[:-len("_compressed.MP4")]
        return stem + ".MP4"
    return basename


async def _get_or_create_default_tenant(db: AsyncSession) -> Tenant:
    result = await db.execute(select(Tenant).where(Tenant.name == DEFAULT_TENANT_NAME))
    tenant = result.scalar_one_or_none()
    if tenant:
        return tenant

    tenant = Tenant(id=uuid.uuid4(), name=DEFAULT_TENANT_NAME, is_active=True)
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)
    return tenant


async def _get_effective_tenant_id(user: User, db: AsyncSession):
    if user.tenant_id:
        return user.tenant_id
    if user.role == "admin":
        default_tenant = await _get_or_create_default_tenant(db)
        user.tenant_id = default_tenant.id
        await db.commit()
        return default_tenant.id
    raise HTTPException(status_code=403, detail="Tenant not assigned")


async def _get_tenant_name_for_user(user: User, db: AsyncSession) -> str:
    """Return tenant name for S3 path (e.g. Farm1). Uses default if needed."""
    tenant_id = await _get_effective_tenant_id(user, db)
    return await _get_tenant_name(db, tenant_id)


def _sanitize_storage_basename(original_name: str, video_id: str) -> str:
    """Build a safe, human-readable storage filename: stem_shortid.ext (e.g. 010421_MVI_0114_short_0_6029_1_91b43afd.mp4)."""
    p = Path(original_name or "video.mp4")
    stem = _safe_filename_part(p.stem)
    ext = p.suffix.lower() or ".mp4"
    if not stem:
        stem = "video"
    short_id = video_id.replace("-", "")[:8] if video_id else uuid.uuid4().hex[:8]
    return f"{stem}_{short_id}{ext}"


def _apply_tenant_filter(query, user: User):
    if user.role == "admin":
        return query
    if not user.tenant_id:
        raise HTTPException(status_code=403, detail="Tenant not assigned")
    return query.where(Video.tenant_id == user.tenant_id)


class ProcessRawVideosRequest(BaseModel):
    keys: List[str]


class ClipRegisterItem(BaseModel):
    raw_video_id: str
    tenant_id: str
    clip_slug: str
    s3_key: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    quality_label: Optional[str] = None
    label: Optional[str] = None
    prob_bad: Optional[float] = None


class BulkRegisterClipsRequest(BaseModel):
    clips: List[ClipRegisterItem]


async def trigger_tleap_pipeline(video_id: str, video_path: str):
    """Trigger T-LEAP pipeline for a video"""
    nc = await get_nats()
    msg = {
        'video_id': video_id,
        'processed_path': video_path
    }
    await nc.publish('video.preprocessed', json.dumps(msg).encode())
    await nc.flush()
    print(f"Triggered T-LEAP pipeline for {video_id}")


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    file_path: str
    file_size: int
    uploaded_at: str
    status: str


class ImportUrlRequest(BaseModel):
    video_url: str
    label: Optional[int] = None
    metadata: Optional[dict] = None


def _guess_extension(url: str, content_type: str | None) -> str:
    url_path = urlparse(url).path
    suffix = Path(url_path).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    if suffix in allowed:
        return suffix
    content_type = (content_type or "").split(";")[0].strip().lower()
    content_map = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
        "video/mp4v-es": ".mp4",
    }
    return content_map.get(content_type, ".mp4")


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_")


def _build_filename_from_url(video_url: str, ext: str, video_id: str) -> str:
    path = Path(urlparse(video_url).path)
    stem = path.stem
    farm = ""
    if len(path.parts) >= 3:
        farm = path.parts[-3]
    elif len(path.parts) >= 2:
        farm = path.parts[-2]
    farm = _safe_filename_part(farm)
    stem = _safe_filename_part(stem)
    if farm and stem:
        base = f"{farm}_{stem}"
    else:
        base = stem or video_id
    return f"{base}{ext}"


@router.get("/storage-config")
async def get_storage_config():
    """Get current storage configuration.

    Returns the active storage backend and its configuration.
    - backend: "local" or "s3"
    - When backend is "local": videos stored in /app/data/videos (EFS in AWS, local volume in dev)
    - When backend is "s3": videos stored in S3 bucket with optional CloudFront CDN
    """
    backend = get_storage_backend()
    return {
        "backend": backend,
        "s3_enabled": is_s3_enabled(),
        "s3_bucket": S3_VIDEOS_BUCKET if is_s3_enabled() else None,
        "cloudfront_enabled": bool(CLOUDFRONT_DOMAIN) and is_s3_enabled(),
        "cloudfront_domain": CLOUDFRONT_DOMAIN if CLOUDFRONT_DOMAIN and is_s3_enabled() else None,
        "local_path": str(VIDEOS_DIR) if backend == "local" else None
    }


def _video_id_from_key(key: str) -> Optional[str]:
    """Extract raw_video_id from S3 key. New: {tenant_name}/raw/{raw_video_id}/original.mp4 or tenants/{tid}/raw/{raw_video_id}/original.mp4. Legacy: .../uuid.mp4 stem."""
    if not key:
        return None
    parts = key.rstrip("/").split("/")
    # Check new format: {tenant_name}/raw/{video_id}/original.mp4
    if len(parts) >= 4 and parts[-1] == "original.mp4" and parts[-3] == "raw":
        return parts[-2]
    # Check legacy tenants format: tenants/{tid}/raw/{video_id}/original.mp4
    if key.startswith(TENANTS_PREFIX) and len(parts) >= 4 and parts[-1] == "original.mp4":
        return parts[-2]
    # Check direct upload format: {tenant_name}/raw/{filename}.mp4 (no video_id folder)
    # For these, we'll use the filename stem as a temporary identifier
    if len(parts) >= 3 and parts[-2] == "raw" and parts[-1].endswith((".mp4", ".MP4", ".avi", ".mov", ".mkv")):
        # This is a direct upload, return None so we handle it specially
        return None
    # Legacy: try to extract from filename
    name = parts[-1] if parts else ""
    stem = Path(name).stem
    if len(stem) == 36 and stem.count("-") == 4:
        return stem
    return None


@router.get("/storage/raw-videos")
async def list_raw_videos(
    include_url: bool = Query(False, description="Include presigned URLs"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID (admin only)"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List raw videos from S3. Uses new key {tenant_name}/raw/; tenant filter applied."""
    import sys
    print(f"[list_raw_videos] ====== API CALLED ======", file=sys.stderr, flush=True)
    print(f"[list_raw_videos] User: {user.id}, Role: {user.role}, Tenant ID param: {tenant_id}", file=sys.stderr, flush=True)
    print(f"[list_raw_videos] User: {user.id}, Role: {user.role}, Tenant ID param: {tenant_id}", flush=True)
    logger.info(f"[list_raw_videos] User: {user.id}, Role: {user.role}, Tenant ID param: {tenant_id}")
    if not is_s3_enabled():
        logger.warning("[list_raw_videos] S3 not enabled")
        return {"items": [], "prefix": "", "tenants": []}
    # Which tenants to list: admin = all (or filtered by tenant_id), else = own tenant
    if user.role == "admin":
        if tenant_id:
            # Filter by specific tenant
            r = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.id == uuid.UUID(tenant_id)))
            row = r.one_or_none()
            if row:
                tenant_info = [(str(row[0]), row[1])]
            else:
                return {"items": [], "prefix": "", "tenants": []}
        else:
            # All tenants
            r = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.is_active == True).order_by(Tenant.name))
            tenant_info = [(str(row[0]), row[1]) for row in r.fetchall()]
    else:
        if not user.tenant_id:
            raise HTTPException(status_code=403, detail="Tenant not assigned")
        r = await db.execute(select(Tenant.name).where(Tenant.id == user.tenant_id))
        tenant_name = r.scalar_one_or_none()
        tenant_info = [(str(user.tenant_id), tenant_name or DEFAULT_TENANT_NAME)]
    
    # Get all tenants for dropdown (admin only)
    all_tenants = []
    if user.role == "admin":
        r_all = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.is_active == True).order_by(Tenant.name))
        all_tenants = [{"id": str(row[0]), "name": row[1]} for row in r_all.fetchall()]
    
    # Build tenant_id to tenant_name mapping from ALL active tenants (for proper tenant name extraction)
    r_all_tenants = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.is_active == True))
    all_tenant_info = [(str(row[0]), row[1]) for row in r_all_tenants.fetchall()]
    tenant_id_to_name = {tid: name for tid, name in all_tenant_info}
    tenant_name_to_id = {name: tid for tid, name in all_tenant_info}
    print(f"[list_raw_videos] Tenant info to query: {tenant_info}", flush=True)
    print(f"[list_raw_videos] All tenants mapping - tenant_name_to_id: {tenant_name_to_id}", flush=True)
    logger.info(f"[list_raw_videos] Tenant info to query: {tenant_info}")
    logger.info(f"[list_raw_videos] All tenants mapping - tenant_name_to_id: {tenant_name_to_id}")
    
    items = []
    for tid, tenant_name in tenant_info:
        tenant_name = tenant_name or DEFAULT_TENANT_NAME
        # Check new path: {tenant_name}/raw/
        prefix = _normalize_prefix(f"{tenant_name}/raw/")
        print(f"[list_raw_videos] Querying S3 with prefix: {prefix}", flush=True)
        prefix_items = _list_s3_objects(prefix)
        print(f"[list_raw_videos] Found {len(prefix_items)} items in {prefix}: {[it['key'] for it in prefix_items]}", flush=True)
        logger.info(f"[list_raw_videos] Querying S3 with prefix: {prefix}")
        logger.info(f"[list_raw_videos] Found {len(prefix_items)} items in {prefix}: {[it['key'] for it in prefix_items]}")
        items.extend(prefix_items)
        # Also check legacy path for backward compatibility
        legacy_prefix = _normalize_prefix(f"{TENANTS_PREFIX}{tid}/raw/")
        print(f"[list_raw_videos] Querying S3 with legacy prefix: {legacy_prefix}", flush=True)
        legacy_items = _list_s3_objects(legacy_prefix)
        print(f"[list_raw_videos] Found {len(legacy_items)} items in {legacy_prefix}: {[it['key'] for it in legacy_items]}", flush=True)
        logger.info(f"[list_raw_videos] Querying S3 with legacy prefix: {legacy_prefix}")
        logger.info(f"[list_raw_videos] Found {len(legacy_items)} items in {legacy_prefix}: {[it['key'] for it in legacy_items]}")
        items.extend(legacy_items)
    print(f"[list_raw_videos] Total S3 items found: {len(items)}, keys: {[it['key'] for it in items]}", flush=True)
    logger.info(f"[list_raw_videos] Total S3 items found: {len(items)}, keys: {[it['key'] for it in items]}")
    if not items:
        print(f"[list_raw_videos] No items found in S3", flush=True)
        logger.warning("[list_raw_videos] No items found in S3")
        return {"items": [], "prefix": _normalize_prefix(TENANTS_PREFIX), "tenants": all_tenants}

    keys_set = {it["key"] for it in items}
    # Extract video_ids, but also handle direct uploads (files directly in raw/ folder)
    video_ids = []
    for item in items:
        vid = _video_id_from_key(item["key"])
        if vid:
            video_ids.append(vid)
    video_ids = list(set(video_ids))
    id_to_info = {}
    key_to_info = {}
    if video_ids:
        result = await db.execute(
            select(Video.id, Video.original_filename, Video.tenant_id, Video.s3_key).where(Video.id.in_(video_ids))
        )
        for row in result:
            id_to_info[str(row.id)] = (row.original_filename, row.tenant_id)
            if row.s3_key and row.s3_key in keys_set:
                key_to_info[row.s3_key] = (str(row.id), row.original_filename, row.tenant_id)
    result_by_key = await db.execute(
        select(Video.id, Video.original_filename, Video.tenant_id, Video.s3_key).where(
            Video.s3_key.isnot(None), Video.s3_key.in_(list(keys_set))
        )
    )
    for row in result_by_key:
        if row.s3_key and row.s3_key in keys_set:
            key_to_info[row.s3_key] = (str(row.id), row.original_filename, row.tenant_id)
        id_to_info[str(row.id)] = (row.original_filename, row.tenant_id)
    
    filtered = []
    for it in items:
        key = it["key"]
        vid = _video_id_from_key(key)
        tenant_name_from_key = None
        # Extract tenant name from key if possible: {tenant_name}/raw/...
        if "/" in key:
            parts = key.split("/")
            potential_tenant_name = parts[0]
            logger.info(f"[list_raw_videos] Processing key: {key}, potential_tenant_name: {potential_tenant_name}")
            logger.info(f"[list_raw_videos] tenant_name_to_id keys: {list(tenant_name_to_id.keys())}")
            # Check if it's a valid tenant name - use tenant_name_to_id mapping which includes all tenants
            if potential_tenant_name in tenant_name_to_id:
                tenant_name_from_key = potential_tenant_name
                logger.info(f"[list_raw_videos] Found tenant_name_from_key via tenant_name_to_id: {tenant_name_from_key}, tenant_id: {tenant_name_to_id[tenant_name_from_key]}")
            # Also check against tenant_id_to_name values
            elif potential_tenant_name in tenant_id_to_name.values():
                tenant_name_from_key = potential_tenant_name
                logger.debug(f"[list_raw_videos] Found tenant_name_from_key via tenant_id_to_name values: {tenant_name_from_key}")
            # Also check if it's not a legacy prefix
            elif not potential_tenant_name.startswith("tenants") and not potential_tenant_name.startswith("raw_videos") and not potential_tenant_name.startswith("short_videos"):
                # Might be a tenant name, check against all known tenant names
                all_tenant_names = list(tenant_id_to_name.values())
                if potential_tenant_name in all_tenant_names:
                    tenant_name_from_key = potential_tenant_name
                    logger.debug(f"[list_raw_videos] Found tenant_name_from_key via all_tenant_names: {tenant_name_from_key}")
                else:
                    logger.debug(f"[list_raw_videos] potential_tenant_name '{potential_tenant_name}' not found in tenant mappings")
        
        # Handle direct uploads: {tenant_name}/raw/{filename}.mp4 (no video_id folder)
        is_direct_upload = False
        if "/" in key:
            parts = key.split("/")
            if len(parts) >= 3 and parts[-2] == "raw" and parts[-1].endswith((".mp4", ".MP4", ".avi", ".mov", ".mkv")) and parts[-1] != "original.mp4":
                is_direct_upload = True
                # Use filename as a temporary identifier
                vid = Path(parts[-1]).stem
                logger.debug(f"[list_raw_videos] Direct upload detected: {key}, tenant_name_from_key: {tenant_name_from_key}")
                if tenant_name_from_key:
                    it["tenant_name"] = tenant_name_from_key
                    if tenant_name_from_key in tenant_name_to_id:
                        it["tenant_id"] = str(tenant_name_to_id[tenant_name_from_key])
                        logger.debug(f"[list_raw_videos] Set tenant_id for direct upload: {it['tenant_id']}")
                it["original_filename"] = parts[-1]
                it["video_id"] = vid  # Use filename stem as temporary ID
                # Don't continue here - we need to apply tenant filter below
        
        if vid and vid in id_to_info:
            orig, tid = id_to_info[vid]
            it["video_id"] = vid
            it["original_filename"] = orig
            it["tenant_id"] = str(tid) if tid else None
            if tid:
                tid_str = str(tid)
                if tid_str in tenant_id_to_name:
                    it["tenant_name"] = tenant_id_to_name[tid_str]
                elif tenant_name_from_key:
                    it["tenant_name"] = tenant_name_from_key
            elif tenant_name_from_key:
                it["tenant_name"] = tenant_name_from_key
                # Try to find tenant_id from tenant_name
                if tenant_name_from_key in tenant_name_to_id:
                    it["tenant_id"] = str(tenant_name_to_id[tenant_name_from_key])
        elif key in key_to_info:
            vid, orig, tid = key_to_info[key]
            it["video_id"] = vid
            it["original_filename"] = orig
            it["tenant_id"] = str(tid) if tid else None
            if tid:
                tid_str = str(tid)
                if tid_str in tenant_id_to_name:
                    it["tenant_name"] = tenant_id_to_name[tid_str]
                elif tenant_name_from_key:
                    it["tenant_name"] = tenant_name_from_key
            elif tenant_name_from_key:
                it["tenant_name"] = tenant_name_from_key
                # Try to find tenant_id from tenant_name
                if tenant_name_from_key in tenant_name_to_id:
                    it["tenant_id"] = str(tenant_name_to_id[tenant_name_from_key])
        else:
            it["video_id"] = vid
            it["original_filename"] = None
            it["tenant_id"] = None
            if tenant_name_from_key:
                it["tenant_name"] = tenant_name_from_key
                # Try to find tenant_id from tenant_name
                if tenant_name_from_key in tenant_name_to_id:
                    it["tenant_id"] = str(tenant_name_to_id[tenant_name_from_key])
        
        # For direct uploads, ensure we have tenant info before filtering
        if is_direct_upload:
            if not it.get("tenant_name") and tenant_name_from_key:
                it["tenant_name"] = tenant_name_from_key
            if not it.get("tenant_id") and tenant_name_from_key and tenant_name_from_key in tenant_name_to_id:
                it["tenant_id"] = str(tenant_name_to_id[tenant_name_from_key])
        
        # Log item state before filtering
        item_tenant_id = it.get("tenant_id")
        item_tenant_name = it.get("tenant_name")
        logger.info(f"[list_raw_videos] Item before filter - key: {key}, tenant_id: {item_tenant_id}, tenant_name: {item_tenant_name}, is_direct_upload: {is_direct_upload}, user.role: {user.role}, filter_tenant_id: {tenant_id}")
        
        # Apply tenant filter for non-admin users
        if user.role != "admin" and user.tenant_id:
            # For non-admin users, only show videos from their tenant
            if it.get("tenant_id") != str(user.tenant_id):
                logger.debug(f"[list_raw_videos] Filtered out (non-admin): item.tenant_id={item_tenant_id}, user.tenant_id={user.tenant_id}")
                continue
        # Apply tenant filter if tenant_id query param is provided (admin only)
        if user.role == "admin" and tenant_id:
            # Admin filtering by specific tenant
            if it.get("tenant_id") != tenant_id:
                logger.debug(f"[list_raw_videos] Filtered out (admin filter): item.tenant_id={item_tenant_id}, filter.tenant_id={tenant_id}")
                continue
        
        print(f"[list_raw_videos] Item passed filter - key: {key}, tenant_id: {item_tenant_id}, tenant_name: {item_tenant_name}", flush=True)
        logger.info(f"[list_raw_videos] Item passed filter - key: {key}, tenant_id: {item_tenant_id}, tenant_name: {item_tenant_name}")
        filtered.append(it)

    key_to_status = _build_s3_key_to_status()
    for it in filtered:
        video_id = it.get("video_id")
        tid = str(it["tenant_id"]) if it.get("tenant_id") else None
        if video_id:
            it["status_info"] = _load_long_video_status(video_id, tid)
        else:
            it["status_info"] = _load_long_video_status(Path(it["filename"]).stem, tid)
        if it["status_info"]["status"] == "not_started" and it["key"] in key_to_status:
            it["status_info"] = key_to_status[it["key"]]
        if include_url:
            it["url"] = _presigned_url_for_key(it["key"])

    print(f"[list_raw_videos] Returning {len(filtered)} filtered items out of {len(items)} total S3 items", flush=True)
    print(f"[list_raw_videos] Filtered items keys: {[it['key'] for it in filtered]}", flush=True)
    logger.info(f"[list_raw_videos] Returning {len(filtered)} filtered items out of {len(items)} total S3 items")
    logger.debug(f"[list_raw_videos] Filtered items keys: {[it['key'] for it in filtered]}")
    return {
        "items": filtered,
        "prefix": _normalize_prefix(TENANTS_PREFIX),
        "tenants": all_tenants,
    }


@router.get("/storage/clips")
async def list_clips(
    include_url: bool = Query(False, description="Include presigned URLs"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID (admin only)"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List clip videos: from DB clip_videos (tenant-filtered) with S3 presigned URL, or fallback to S3 list by {tenant_name}/clips/."""
    if not is_s3_enabled():
        return {"items": [], "prefix": "", "tenants": []}
    # Get tenant info
    if user.role == "admin":
        if tenant_id:
            # Filter by specific tenant
            r = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.id == uuid.UUID(tenant_id)))
            row = r.one_or_none()
            if row:
                tenant_info = [(row[0], row[1])]
            else:
                return {"items": [], "prefix": "", "tenants": []}
        else:
            # All tenants
            r = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.is_active == True).order_by(Tenant.name))
            tenant_info = [(row[0], row[1]) for row in r.fetchall()]
    else:
        if not user.tenant_id:
            raise HTTPException(status_code=403, detail="Tenant not assigned")
        r = await db.execute(select(Tenant.name).where(Tenant.id == user.tenant_id))
        tenant_name = r.scalar_one_or_none()
        tenant_info = [(user.tenant_id, tenant_name or DEFAULT_TENANT_NAME)]
    
    # Get all tenants for dropdown (admin only)
    all_tenants = []
    if user.role == "admin":
        r_all = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.is_active == True).order_by(Tenant.name))
        all_tenants = [{"id": str(row[0]), "name": row[1]} for row in r_all.fetchall()]
    
    tenant_ids = [tid for tid, _ in tenant_info]
    tenant_id_to_name = {tid: name for tid, name in tenant_info}
    # Prefer DB clip_videos for tenant-scoped list, but verify S3 files exist
    q = select(ClipVideo).where(ClipVideo.tenant_id.in_(tenant_ids)).order_by(ClipVideo.raw_video_id, ClipVideo.clip_slug)
    result = await db.execute(q)
    clips = result.scalars().all()
    if clips:
        # Build mapping from raw_video_id -> human-readable name (original_filename or filename)
        raw_ids = list({c.raw_video_id for c in clips if c.raw_video_id})
        raw_id_to_name: Dict[str, Optional[str]] = {}
        if raw_ids:
            r_videos = await db.execute(
                select(Video.id, Video.original_filename, Video.filename).where(Video.id.in_(raw_ids))
            )
            for vid_id, original_filename, filename in r_videos:
                display_name = original_filename or filename
                if display_name:
                    raw_id_to_name[str(vid_id)] = display_name
            # Fallback to pipeline summary file when DB has no name
            for vid_id in raw_ids:
                if not raw_id_to_name.get(str(vid_id)):
                    from_summary = _parent_display_name_from_summary(str(vid_id))
                    if from_summary:
                        raw_id_to_name[str(vid_id)] = from_summary

        # Check which S3 keys actually exist
        s3 = get_s3_client()
        existing_keys = set()
        for c in clips:
            if c.s3_key:
                try:
                    s3.head_object(Bucket=S3_VIDEOS_BUCKET, Key=c.s3_key)
                    existing_keys.add(c.s3_key)
                except Exception:
                    # File doesn't exist in S3, skip it
                    pass
        
        items = []
        for c in clips:
            # Only include clips that exist in S3
            if not c.s3_key or c.s3_key not in existing_keys:
                continue
            
            # Get tenant name for this clip
            tenant_name = None
            if c.tenant_id:
                tid_str = str(c.tenant_id)
                if tid_str in tenant_id_to_name:
                    tenant_name = tenant_id_to_name[tid_str]
                else:
                    # Fallback: extract from s3_key
                    if "/" in c.s3_key:
                        parts = c.s3_key.split("/")
                        if parts[0] in tenant_id_to_name.values():
                            tenant_name = parts[0]
            
            parent_video_name = raw_id_to_name.get(c.raw_video_id)
            it = {
                "key": c.s3_key,
                "filename": Path(c.s3_key).name,
                "size": 0,
                "last_modified": c.created_at.isoformat() if c.created_at else None,
                "label": c.label,
                "prob_bad": c.prob_bad,
                "parent_video_id": c.raw_video_id,
                "parent_video_name": parent_video_name,
                "clip_slug": c.clip_slug,
                "clip_id": str(c.id),
                "tenant_id": str(c.tenant_id) if c.tenant_id else None,
                "tenant_name": tenant_name,
            }
            if include_url:
                it["url"] = _presigned_url_for_key(c.s3_key)
            items.append(it)
        
        # Only return DB items if we found valid clips
        if items:
            return {"items": items, "prefix": "", "source": "db", "tenants": all_tenants}
    # Fallback: list S3 by tenant name prefix
    items = []
    for tid, tenant_name in tenant_info:
        tenant_name = tenant_name or DEFAULT_TENANT_NAME
        # Check new path: {tenant_name}/clips/
        prefix_clips = _normalize_prefix(f"{tenant_name}/clips/")
        items.extend(_list_s3_objects(prefix_clips))
        # Also check legacy paths for backward compatibility
        legacy_prefix_clips = _normalize_prefix(f"{TENANTS_PREFIX}{str(tid)}/clips/")
        items.extend(_list_s3_objects(legacy_prefix_clips))
        legacy_prefix_short = _normalize_prefix(f"{TENANTS_PREFIX}{str(tid)}/short/")
        items.extend(_list_s3_objects(legacy_prefix_short))
    label_map = _build_clips_label_map()
    tenant_name_to_id = {name: tid for tid, name in tenant_info}
    filtered_items = []
    parent_ids: Set[str] = set()
    for item in items:
        key = item.get("key") or ""
        parts = key.rstrip("/").split("/")
        # Extract tenant name from key: {tenant_name}/clips/...
        if len(parts) >= 3:
            potential_tenant_name = parts[0]
            # Check if it's a valid tenant name (not legacy prefixes)
            if potential_tenant_name not in ["tenants", "raw_videos", "short_videos", "clips_videos"]:
                if potential_tenant_name in tenant_id_to_name.values():
                    item["tenant_name"] = potential_tenant_name
                    # Also try to get tenant_id
                    if potential_tenant_name in tenant_name_to_id:
                        item["tenant_id"] = str(tenant_name_to_id[potential_tenant_name])
                elif potential_tenant_name in tenant_name_to_id:
                    # Found tenant_id from name
                    item["tenant_id"] = str(tenant_name_to_id[potential_tenant_name])
                    item["tenant_name"] = potential_tenant_name

        # Derive clip_slug from filename if not provided (e.g. clip_1, clip_2, ...)
        if "clip_slug" not in item or not item.get("clip_slug"):
            try:
                item["clip_slug"] = Path(key).stem
            except Exception:
                pass

        # Derive parent_video_id from key structure: Farm1/clips/{video_id}/clip_1.mp4
        # parts = ["Farm1", "clips", "{video_id}", "clip_1.mp4"]
        # parts[-2] = video_id, parts[-1] = clip_1.mp4
        parent_video_id = None
        if len(parts) >= 4:
            # Format: {tenant_name}/clips/{video_id}/clip_1.mp4
            if parts[-3] == "clips":
                parent_video_id = parts[-2]
            # Format: tenants/{tenant_id}/clips/{video_id}/clip_1.mp4
            elif len(parts) >= 5 and parts[-3] == "clips" and parts[0] == "tenants":
                parent_video_id = parts[-2]
            # Format: tenants/{tenant_id}/short/{video_id}/clip_1.mp4
            elif len(parts) >= 5 and parts[-3] == "short" and parts[0] == "tenants":
                parent_video_id = parts[-2]
        
        # Fallback: try to get from label_map if key parsing failed
        if not parent_video_id:
            parent_video_id = label_map.get(item["filename"], {}).get("parent_video_id")

        if parent_video_id:
            item["parent_video_id"] = str(parent_video_id)
            parent_ids.add(str(parent_video_id))

        info = label_map.get(item["filename"])
        if info:
            item["label"] = info.get("label")
            item["prob_bad"] = info.get("prob_bad")
        
        # Apply tenant filter if tenant_id query param is provided (admin only)
        if user.role == "admin" and tenant_id:
            item_tid = item.get("tenant_id")
            if item_tid != tenant_id:
                continue  # Skip this item
        
        filtered_items.append(item)

    items = filtered_items

    # Enrich with human-readable parent_video_name using DB lookup, then summary file fallback
    # Goal: MVI_0115_short_0.MP4 -> MVI_0115_short_0_clip_1.MP4
    if parent_ids:
        raw_id_to_name: Dict[str, Optional[str]] = {}
        try:
            result_videos = await db.execute(
                select(Video.id, Video.original_filename, Video.filename).where(Video.id.in_(list(parent_ids)))
            )
            for vid_id, original_filename, filename in result_videos:
                display_name = original_filename or filename
                if display_name:
                    raw_id_to_name[str(vid_id)] = display_name
        except Exception:
            pass
        # Fallback to pipeline summary file when DB has no name
        for pid in parent_ids:
            if not raw_id_to_name.get(pid):
                from_summary = _parent_display_name_from_summary(pid)
                if from_summary:
                    raw_id_to_name[pid] = from_summary

        for item in items:
            pid = item.get("parent_video_id")
            if pid and pid in raw_id_to_name:
                item["parent_video_name"] = raw_id_to_name[pid]

    if include_url:
        for item in items:
            item["url"] = _presigned_url_for_key(item["key"])
    return {"items": items, "prefix": _normalize_prefix(TENANTS_PREFIX), "source": "s3", "tenants": all_tenants}


@router.get("/storage/clips/url")
async def get_clip_url(
    key: str = Query(..., description="S3 key of the clip video"),
    user: User = Depends(get_current_user),
):
    """Get a presigned URL for a single clip video (for View -> same video page)."""
    # Accept new path: {tenant_name}/clips/, legacy paths: tenants/{tenant_id}/clips/ or tenants/{tenant_id}/short/
    valid = "/clips/" in key or "/short/" in key
    if not valid:
        valid = key.startswith(_normalize_prefix(CLIPS_VIDEOS_PREFIX)) or key.startswith(_normalize_prefix(SHORT_VIDEOS_PREFIX))
    if not valid:
        raise HTTPException(status_code=400, detail="Invalid clip video key")
    url = _presigned_url_for_key(key)
    if not url:
        raise HTTPException(status_code=404, detail="Could not generate URL")
    return {"url": url, "key": key}


@router.post("/storage/clips/register")
async def bulk_register_clips(
    payload: BulkRegisterClipsRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register short clips into clip_videos (e.g. after pipeline uploads to S3). Admin/researcher/tenant_admin only."""
    if user.role not in {"admin", "tenant_admin", "researcher"}:
        raise HTTPException(status_code=403, detail="Access denied")
    if not payload.clips:
        return {"registered": 0, "errors": []}
    errors = []
    registered = 0
    for item in payload.clips:
        if user.role != "admin" and user.tenant_id and str(user.tenant_id) != item.tenant_id:
            errors.append({"clip_slug": item.clip_slug, "error": "tenant_id does not match"})
            continue
        try:
            tid = uuid.UUID(item.tenant_id)
        except (ValueError, TypeError):
            errors.append({"clip_slug": item.clip_slug, "error": "invalid tenant_id"})
            continue
        existing = await db.execute(
            select(ClipVideo).where(ClipVideo.s3_key == item.s3_key)
        )
        if existing.scalar_one_or_none():
            continue
        clip = ClipVideo(
            tenant_id=tid,
            raw_video_id=item.raw_video_id,
            clip_slug=item.clip_slug,
            s3_key=item.s3_key,
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            quality_label=item.quality_label,
            label=item.label,
            prob_bad=item.prob_bad,
        )
        db.add(clip)
        registered += 1
    await db.commit()
    return {"registered": registered, "errors": errors}


@router.post("/storage/raw-videos/process")
async def process_raw_videos(
    payload: ProcessRawVideosRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger preprocessing for raw videos (do NOT download synchronously). Viewer cannot process."""
    logger.info("process_raw_videos called with keys=%s", payload.keys)
    if user.role not in {"admin", "tenant_admin", "researcher"}:
        raise HTTPException(status_code=403, detail="Access denied")
    if not payload.keys:
        raise HTTPException(status_code=400, detail="No keys provided")
    if not is_s3_enabled():
        logger.warning("process_raw_videos: S3 not configured, rejecting")
        raise HTTPException(status_code=400, detail="S3 storage not configured")

    new_prefix = TENANTS_PREFIX
    long_prefix_legacy = _normalize_prefix(RAW_VIDEOS_PREFIX)
    LONG_VIDEO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    key_to_video_id = {}
    key_to_tenant_id = {}
    if payload.keys:
        res = await db.execute(select(Video.id, Video.s3_key, Video.tenant_id).where(Video.s3_key.in_(payload.keys)))
        for row in res:
            if row.s3_key:
                key_to_video_id[row.s3_key] = str(row.id)
                if row.tenant_id:
                    key_to_tenant_id[row.s3_key] = str(row.tenant_id)

    processed = []
    errors = []
    subject = _get_subject("video_uploaded", "video.uploaded")
    logger.info("process_raw_videos: NATS subject=%s", subject)
    nc = await get_nats()

    for key in payload.keys:
        parts = key.rstrip("/").split("/")
        tenant_name = None
        is_new_key = key.startswith(new_prefix)  # tenants/{tid}/raw/{video_id}/original.mp4
        is_tenant_name_video_id_key = (
            len(parts) >= 4 and
            parts[-3] == "raw" and
            parts[-1] == "original.mp4"
        )  # {tenant_name}/raw/{video_id}/original.mp4
        is_tenant_name_key = len(parts) >= 3 and parts[-2] == "raw" and parts[-1].endswith((".mp4", ".MP4", ".avi", ".mov", ".mkv"))  # {tenant_name}/raw/{filename}.mp4
        is_legacy_key = key.startswith(long_prefix_legacy)  # raw_videos/...
        
        if not is_new_key and not is_tenant_name_video_id_key and not is_tenant_name_key and not is_legacy_key:
            err_msg = "Invalid raw video key (expected tenants/.../raw/{video_id}/original.mp4, {tenant_name}/raw/{video_id}/original.mp4, {tenant_name}/raw/{filename}.mp4, or raw_videos/...)"
            errors.append({"key": key, "error": err_msg})
            logger.warning("process_raw_videos: key rejected %s: %s", key, err_msg)
            continue
        
        if is_new_key:
            # Format: tenants/{tenant_id}/raw/{video_id}/original.mp4
            if len(parts) < 4 or parts[-1] != "original.mp4":
                errors.append({"key": key, "error": "Invalid tenants/.../raw/ key format"})
                continue
            video_id = parts[-2]
            tenant_id = key_to_tenant_id.get(key) or parts[1]
            filename = "original.mp4"
        elif is_tenant_name_video_id_key:
            # Format: {tenant_name}/raw/{video_id}/original.mp4
            tenant_name_from_path = parts[0]
            video_id = parts[-2]
            filename = "original.mp4"
            # Resolve tenant_id from tenant name (case-insensitive so S3 path matches DB)
            r_tenant = await db.execute(
                select(Tenant.id, Tenant.name).where(func.lower(Tenant.name) == tenant_name_from_path.lower())
            )
            row_tenant = r_tenant.one_or_none()
            if not row_tenant:
                err_msg = f"Tenant '{tenant_name_from_path}' not found. Create a tenant with this name in User Management (or use an existing tenant folder name)."
                errors.append({"key": key, "error": err_msg})
                logger.warning("process_raw_videos: key rejected %s: %s", key, err_msg)
                continue
            tenant_id = str(row_tenant[0])
            tenant_name = (row_tenant[1] or tenant_name_from_path).strip()
        elif is_tenant_name_key:
            # Format: {tenant_name}/raw/{filename}.mp4 (direct upload)
            tenant_name_from_path = parts[0]  # Keep the tenant_name from path
            filename = parts[-1]
            # Get tenant_id from tenant_name (case-insensitive so S3 path matches DB)
            r_tenant = await db.execute(
                select(Tenant.id, Tenant.name).where(func.lower(Tenant.name) == tenant_name_from_path.lower())
            )
            row_tenant = r_tenant.one_or_none()
            if not row_tenant:
                err_msg = f"Tenant '{tenant_name_from_path}' not found. Create a tenant with this name in User Management (or use an existing tenant folder name)."
                errors.append({"key": key, "error": err_msg})
                logger.warning("process_raw_videos: key rejected %s: %s", key, err_msg)
                continue
            tenant_id = str(row_tenant[0])
            tenant_name = (row_tenant[1] or tenant_name_from_path).strip()  # Use DB name for consistency
            # Use filename stem as video_id (or generate new one if not in DB)
            video_id = key_to_video_id.get(key) or Path(filename).stem
            if len(video_id) != 36 or video_id.count("-") != 4:
                # Not a UUID, generate a new one
                video_id = str(uuid.uuid4())
        else:
            # Legacy format: raw_videos/...
            filename = Path(key).name
            if not filename:
                errors.append({"key": key, "error": "Invalid filename"})
                continue
            video_id = key_to_video_id.get(key) or (Path(filename).stem if len(Path(filename).stem) == 36 else str(uuid.uuid4()))
            tenant_id = key_to_tenant_id.get(key)
            tenant_name = None  # Will be looked up from tenant_id below
        
        if user.role != "admin" and user.tenant_id and key_to_tenant_id.get(key) != str(user.tenant_id):
            errors.append({"key": key, "error": "Not your tenant"})
            logger.warning("process_raw_videos: key rejected %s: Not your tenant", key)
            continue

        # Only lookup tenant_name from tenant_id if we don't already have it (from path)
        if tenant_id and not tenant_name:
            tenant_name = await _get_tenant_name(db, tenant_id)
        message = {
            "video_id": video_id,
            "file_path": "",
            "filename": filename,
            "file_size": 0,
            "uploaded_at": datetime.utcnow().isoformat(),
            "metadata": {"source": "s3_long_videos", "s3_key": key},
        }
        if tenant_id:
            message["metadata"]["tenant_id"] = tenant_id
        if tenant_name:
            message["metadata"]["tenant_name"] = tenant_name
        logger.info("process_raw_videos: publishing to %s video_id=%s s3_key=%s", subject, video_id, key)
        await nc.publish(subject, json.dumps(message).encode())
        await nc.flush()
        queued_path = LONG_VIDEO_RESULTS_DIR / f"{video_id}_queued.json"
        try:
            with queued_path.open("w", encoding="utf-8") as f:
                json.dump({"status": "queued", "reason": "", "queued_at": datetime.utcnow().isoformat(), "s3_key": key}, f, indent=2)
        except Exception:
            pass
        processed.append({"key": key, "video_id": video_id, "local_path": None})

    logger.info("process_raw_videos: done published=%d errors=%d", len(processed), len(errors))
    return {"processed": processed, "errors": errors, "count": len(processed)}


@router.post("/upload-url")
async def get_upload_url(
    filename: str = Query(..., description="Original filename"),
    content_type: str = Query("video/mp4", description="Content type"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a pre-signed URL for direct S3 upload. Uses tenants/{tenant_id}/raw/{video_id}/original.mp4. Viewer cannot upload."""
    if user.role == "viewer":
        raise HTTPException(status_code=403, detail="Viewer cannot upload")
    if not is_s3_enabled():
        raise HTTPException(status_code=400, detail="S3 storage not configured. Use /upload endpoint instead.")

    tenant_id = await _get_effective_tenant_id(user, db)
    tenant_name = await _get_tenant_name(db, tenant_id)
    s3 = get_s3_client()
    video_id = str(uuid.uuid4())
    s3_key = _raw_s3_key(tenant_name, video_id)

    try:
        # Generate pre-signed POST URL (better for browser uploads)
        presigned_post = s3.generate_presigned_post(
            Bucket=S3_VIDEOS_BUCKET,
            Key=s3_key,
            Fields={
                "Content-Type": content_type,
            },
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, 5 * 1024 * 1024 * 1024],  # Max 5GB
            ],
            ExpiresIn=3600  # 1 hour
        )

        return {
            "video_id": video_id,
            "upload_url": presigned_post["url"],
            "upload_fields": presigned_post["fields"],
            "s3_key": s3_key,
            "expires_in": 3600
        }
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")


@router.post("/confirm-upload")
async def confirm_upload(
    video_id: str = Query(...),
    s3_key: str = Query(...),
    label: Optional[int] = Query(None),
    original_filename: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Confirm that a video upload to S3 is complete. Creates a database record. Viewer cannot upload."""
    if user.role == "viewer":
        raise HTTPException(status_code=403, detail="Viewer cannot upload")
    if not is_s3_enabled():
        raise HTTPException(status_code=400, detail="S3 storage not configured")

    s3 = get_s3_client()
    if not s3_key.startswith(TENANTS_PREFIX) or "/raw/" not in s3_key:
        raise HTTPException(status_code=403, detail="Invalid raw video key (expected tenants/{tid}/raw/...)")
    parts = s3_key.rstrip("/").split("/")
    if len(parts) < 4 or parts[-1] != "original.mp4":
        raise HTTPException(status_code=403, detail="Invalid key format")
    key_tenant_id = parts[1]
    if user.role != "admin" and user.tenant_id and str(user.tenant_id) != key_tenant_id:
        raise HTTPException(status_code=403, detail="Key does not belong to your tenant")

    try:
        head = s3.head_object(Bucket=S3_VIDEOS_BUCKET, Key=s3_key)
        file_size = head["ContentLength"]
    except ClientError:
        raise HTTPException(status_code=404, detail="Video not found in S3")

    filename = parts[-1]
    uploaded_at = datetime.utcnow()
    try:
        tenant_uuid = uuid.UUID(key_tenant_id) if key_tenant_id else user.tenant_id
    except (ValueError, TypeError):
        tenant_uuid = user.tenant_id

    video_record = Video(
        id=video_id,
        filename=filename,
        original_filename=original_filename,
        file_size=file_size,
        storage_backend="s3",
        s3_key=s3_key,
        tenant_id=tenant_uuid,
        uploaded_by=user.id,
        label=label if label is not None and label in [0, 1] else None,
        label_confidence="certain" if label is not None and label in [0, 1] else None,
        status="uploaded",
        uploaded_at=uploaded_at
    )
    db.add(video_record)
    await db.commit()

    return {
        "video_id": video_id,
        "s3_key": s3_key,
        "file_size": file_size,
        "uploaded_at": uploaded_at.isoformat(),
        "label": label if label is not None and label in [0, 1] else None,
        "label_saved": label is not None and label in [0, 1],
        "stream_url": get_video_stream_url(video_id, s3_key)
    }


def get_video_stream_url(video_id: str, s3_key: str = None) -> str:
    """Get the streaming URL for a video.

    If CloudFront is configured, returns CloudFront URL.
    Otherwise returns S3 pre-signed URL.
    """
    if not s3_key:
        s3_key = f"raw/{video_id}.mp4"

    if CLOUDFRONT_DOMAIN:
        return f"https://{CLOUDFRONT_DOMAIN}/{s3_key}"
    elif is_s3_enabled():
        s3 = get_s3_client()
        try:
            return s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_VIDEOS_BUCKET, "Key": s3_key},
                ExpiresIn=3600  # 1 hour
            )
        except ClientError:
            return None
    return None


@router.get("/{video_id}/stream-url")
async def get_stream_url(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the streaming URL for a video.

    Returns CloudFront URL if configured, or falls back to S3 pre-signed URL.
    """
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.storage_backend != "s3" or not is_s3_enabled():
        return {
            "video_id": video_id,
            "stream_url": f"/api/videos/{video_id}/stream",
            "source": "local"
        }

    s3_key = video.s3_key
    if not s3_key:
        raise HTTPException(status_code=404, detail="Video not found in S3")

    stream_url = get_video_stream_url(video_id, s3_key)

    return {
        "video_id": video_id,
        "stream_url": stream_url,
        "s3_key": s3_key,
        "source": "cloudfront" if CLOUDFRONT_DOMAIN else "s3"
    }


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    label: Optional[int] = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload a video file with optional label (0=sound, 1=lame). Viewer cannot upload."""
    if user.role == "viewer":
        raise HTTPException(status_code=403, detail="Viewer cannot upload")
    raw_name = (file.filename or "").strip()
    original_filename = raw_name if raw_name else None

    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = Path(raw_name or "video.mp4").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate video ID
    video_id = str(uuid.uuid4())
    uploaded_at = datetime.utcnow()

    if is_s3_enabled():
        tenant_id = await _get_effective_tenant_id(user, db)
        tenant_name = await _get_tenant_name(db, tenant_id)
        s3 = get_s3_client()
        s3_key = _raw_s3_key(tenant_name, video_id)
        filename = "original.mp4"

        try:
            # Read file content
            file_content = await file.read()
            file_size = len(file_content)
            content_type = file.content_type or "video/mp4"

            # Arbutus requires explicit Content-Length; boto3 can omit it. Use presigned PUT + httpx.
            presigned_url = s3.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": S3_VIDEOS_BUCKET,
                    "Key": s3_key,
                    "ContentType": content_type,
                },
                ExpiresIn=3600,
            )
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.put(
                    presigned_url,
                    content=file_content,
                    headers={
                        "Content-Type": content_type,
                        "Content-Length": str(file_size),
                    },
                )
            if r.status_code >= 400:
                raise ClientError(
                    {"Error": {"Code": "HttpError", "Message": f"PUT {r.status_code}: {r.text[:200]}"}},
                    "PutObject",
                )

            video_record = Video(
                id=video_id,
                filename=filename,
                original_filename=original_filename,
                file_size=file_size,
                storage_backend="s3",
                s3_key=s3_key,
                tenant_id=tenant_id,
                uploaded_by=user.id,
                label=label if label is not None and label in [0, 1] else None,
                label_confidence="certain" if label is not None and label in [0, 1] else None,
                status="uploaded",
                uploaded_at=uploaded_at,
            )
            db.add(video_record)
            await db.commit()

            return {
                "video_id": video_id,
                "filename": original_filename or filename,
                "s3_key": s3_key,
                "file_size": file_size,
                "uploaded_at": uploaded_at.isoformat(),
                "label": label if label is not None and label in [0, 1] else None,
                "label_saved": label is not None and label in [0, 1],
                "storage": "s3",
                "stream_url": get_video_stream_url(video_id, s3_key)
            }
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

    # Fall back to local storage (EFS)
    tenant_id = await _get_effective_tenant_id(user, db)
    filename = _sanitize_storage_basename(raw_name or "video.mp4", video_id)
    tenant_dir = VIDEOS_DIR / "tenant" / str(tenant_id)
    file_path = tenant_dir / filename
    tenant_dir.mkdir(parents=True, exist_ok=True)

    file_size = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                file_size += len(chunk)

        # Create database record
        video_record = Video(
            id=video_id,
            filename=filename,
            original_filename=original_filename,
            file_size=file_size,
            storage_backend="local",
            file_path=str(file_path),
            tenant_id=tenant_id,
            uploaded_by=user.id,
            label=label if label is not None and label in [0, 1] else None,
            label_confidence="certain" if label is not None and label in [0, 1] else None,
            status="uploaded",
            uploaded_at=uploaded_at
        )
        db.add(video_record)
        await db.commit()

        return {
            "video_id": video_id,
            "filename": original_filename or filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "uploaded_at": uploaded_at.isoformat(),
            "label": label if label is not None and label in [0, 1] else None,
            "label_saved": label is not None and label in [0, 1],
            "storage": "local"
        }
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/import-url")
async def import_video_from_url(
    payload: ImportUrlRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Import a video from a public URL and trigger processing."""
    video_url = payload.video_url.strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="video_url is required")

    parsed = urlparse(video_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="video_url must be http(s)")

    tenant_id = await _get_effective_tenant_id(user, db)
    tenant_dir = VIDEOS_DIR / "tenant" / str(tenant_id)
    tenant_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(uuid.uuid4())
    uploaded_at = datetime.utcnow()

    timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
    max_bytes = int(os.getenv("MAX_URL_VIDEO_BYTES", str(5 * 1024 * 1024 * 1024)))

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            head_resp = None
            try:
                head_resp = await client.head(video_url)
            except Exception:
                head_resp = None

            content_type = head_resp.headers.get("content-type") if head_resp else None
            content_length = None
            if head_resp:
                try:
                    content_length = int(head_resp.headers.get("content-length", "0"))
                except ValueError:
                    content_length = None
            if content_length and content_length > max_bytes:
                raise HTTPException(status_code=413, detail="Video exceeds maximum allowed size")

            ext = _guess_extension(video_url, content_type)
            filename = _build_filename_from_url(video_url, ext, video_id)
            file_path = tenant_dir / filename
            if file_path.exists():
                filename = f"{Path(filename).stem}_{video_id[:8]}{ext}"
                file_path = tenant_dir / filename

            resp = await client.get(video_url)
            resp.raise_for_status()
            try:
                resp_length = int(resp.headers.get("content-length", "0"))
            except ValueError:
                resp_length = 0
            if resp_length and resp_length > max_bytes:
                raise HTTPException(status_code=413, detail="Video exceeds maximum allowed size")

            file_size = 0
            with open(file_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file_size += len(chunk)
                    if file_size > max_bytes:
                        raise HTTPException(status_code=413, detail="Video exceeds maximum allowed size")
                    f.write(chunk)

        storage_backend = "local"
        s3_key = None
        if is_s3_enabled():
            s3 = get_s3_client()
            s3_key = f"{_normalize_prefix(RAW_VIDEOS_PREFIX)}{filename}"
            try:
                ct = content_type or "video/mp4"
                presigned_url = s3.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": S3_VIDEOS_BUCKET,
                        "Key": s3_key,
                        "ContentType": ct,
                    },
                    ExpiresIn=3600,
                )
                file_content = Path(file_path).read_bytes()
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.put(
                        presigned_url,
                        content=file_content,
                        headers={
                            "Content-Type": ct,
                            "Content-Length": str(file_size),
                        },
                    )
                if r.status_code >= 400:
                    raise ClientError(
                        {"Error": {"Code": "HttpError", "Message": f"PUT {r.status_code}: {r.text[:200]}"}},
                        "PutObject",
                    )
                storage_backend = "s3"
            except ClientError as e:
                raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

        video_record = Video(
            id=video_id,
            filename=filename,
            original_filename=Path(urlparse(video_url).path).name or filename,
            file_size=file_size,
            storage_backend=storage_backend,
            s3_key=s3_key,
            file_path=str(file_path),
            tenant_id=user.tenant_id,
            uploaded_by=user.id,
            label=payload.label if payload.label is not None and payload.label in [0, 1] else None,
            label_confidence="certain" if payload.label is not None and payload.label in [0, 1] else None,
            status="uploaded",
            uploaded_at=uploaded_at
        )
        db.add(video_record)
        await db.commit()

        tenant_name = await _get_tenant_name(db, user.tenant_id) if user.tenant_id else None
        nc = await get_nats()
        subject = _get_subject("video_uploaded", "video.uploaded")
        message = {
            "video_id": video_id,
            "file_path": str(file_path),
            "filename": filename,
            "file_size": file_size,
            "uploaded_at": uploaded_at.isoformat(),
            "metadata": payload.metadata or {},
            "source": "url",
        }
        if s3_key:
            message["s3_key"] = s3_key
        if user.tenant_id:
            message["metadata"]["tenant_id"] = str(user.tenant_id)
        if tenant_name:
            message["metadata"]["tenant_name"] = tenant_name
        await nc.publish(subject, json.dumps(message).encode())
        await nc.flush()

        response = {
            "video_id": video_id,
            "filename": filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "uploaded_at": uploaded_at.isoformat(),
            "label": payload.label if payload.label is not None and payload.label in [0, 1] else None,
            "label_saved": payload.label is not None and payload.label in [0, 1],
            "storage": storage_backend
        }
        if s3_key:
            response["s3_key"] = s3_key
            response["stream_url"] = get_video_stream_url(video_id, s3_key)
        return response
    except HTTPException:
        raise
    except Exception as e:
        if "file_path" in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/{video_id}")
async def get_video(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get video information from database"""
    # Query video from database
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Check for analysis results (local files for now)
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    has_analysis = video.has_analysis or fusion_file.exists()

    # Check for annotated video
    annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
    has_annotated = video.has_annotated or annotated_file.exists()

    # Build stream URL
    stream_url = None
    if video.storage_backend == "s3" and video.s3_key:
        stream_url = get_video_stream_url(video_id, video.s3_key)
    elif video.file_path:
        stream_url = f"/api/videos/{video_id}/stream"

    return {
        "video_id": video_id,
        "filename": video.filename,
        "original_filename": video.original_filename,
        "file_size": video.file_size,
        "storage": video.storage_backend,
        "s3_key": video.s3_key,
        "file_path": video.file_path,
        "tenant_id": str(video.tenant_id) if video.tenant_id else None,
        "stream_url": stream_url,
        "has_analysis": has_analysis,
        "has_annotated": has_annotated,
        "label": video.label,
        "label_confidence": video.label_confidence,
        "status": video.status,
        "uploaded_at": video.uploaded_at.isoformat() if video.uploaded_at else None,
        "processed_at": video.processed_at.isoformat() if video.processed_at else None,
        "metadata": {
            "fps": video.fps,
            "frame_count": video.frame_count,
            "width": video.width,
            "height": video.height,
            "duration": video.duration
        }
    }


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Stream the original video file"""
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.storage_backend == "s3":
        raise HTTPException(status_code=400, detail="Video is stored in S3")
    if not video.file_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    file_path = Path(video.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska"
    }
    media_type = media_types.get(suffix, "video/mp4")
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name
    )


@router.get("/{video_id}/annotated")
async def stream_annotated_video(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Stream the annotated video file"""
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
    
    if not annotated_file.exists():
        raise HTTPException(
            status_code=404, 
            detail="Annotated video not found. Trigger annotation first."
        )
    
    return FileResponse(
        path=str(annotated_file),
        media_type="video/mp4",
        filename=f"{video_id}_annotated.mp4"
    )


@router.get("/{video_id}/frame/{frame_num}")
async def get_frame(
    video_id: str,
    frame_num: int,
    annotated: bool = False,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific frame from the video as an image"""
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Choose source file
    if annotated:
        video_path = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Annotated video not found")
    else:
        if video.storage_backend == "s3":
            raise HTTPException(status_code=400, detail="Video is stored in S3")
        if not video.file_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        video_path = Path(video.file_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
    
    # Extract frame
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_num < 0 or frame_num >= total_frames:
        cap.release()
        raise HTTPException(status_code=400, detail=f"Frame number must be 0-{total_frames-1}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to read frame")
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )


@router.post("/{video_id}/annotate")
async def trigger_annotation(
    video_id: str,
    include_yolo: bool = True,
    include_pose: bool = True,
    show_confidence: bool = True,
    show_labels: bool = True,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Trigger annotation rendering for a video.

    If pose data doesn't exist and include_pose=True, triggers T-LEAP pipeline first.
    """
    # Check video exists in database
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Determine video path or S3 key
    video_path = None
    s3_key = None

    if video.storage_backend == "s3" and video.s3_key:
        s3_key = video.s3_key
        # For S3 videos, we pass the S3 key to the annotation service
        video_path = f"s3://{S3_VIDEOS_BUCKET}/{s3_key}"
    elif video.file_path:
        video_path = video.file_path
    else:
        # Fallback to legacy path
        video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
        if video_files:
            video_path = str(video_files[0])
        else:
            raise HTTPException(status_code=404, detail="Video file not found")
    
    # Check if T-LEAP pose data exists (don't wait for it - proceed with YOLO only if not available)
    tleap_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    pose_available = tleap_file.exists()

    if include_pose and not pose_available:
        print(f"Pose data not found for {video_id}, proceeding with YOLO annotations only")
        # Override include_pose to false since data doesn't exist
        include_pose = False

    # Call annotation renderer service
    try:
        request_data = {
            "video_id": video_id,
            "include_yolo": include_yolo,
            "include_pose": include_pose,
            "show_confidence": show_confidence,
            "show_labels": show_labels,
            "video_path": video_path,
        }
        # Add S3 info if video is stored in S3
        if s3_key:
            request_data["s3_bucket"] = S3_VIDEOS_BUCKET
            request_data["s3_key"] = s3_key

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ANNOTATION_RENDERER_URL}/render",
                json=request_data,
                timeout=30.0
            )
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Annotation renderer service unavailable"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{video_id}/annotation-status")
async def get_annotation_status(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get annotation rendering status"""
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ANNOTATION_RENDERER_URL}/status/{video_id}",
                timeout=10.0
            )
            return response.json()
    except httpx.ConnectError:
        # Service unavailable, check if file exists
        annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        if annotated_file.exists():
            return {
                "video_id": video_id,
                "status": "completed",
                "progress": 1.0,
                "output_path": str(annotated_file)
            }
        return {
            "video_id": video_id,
            "status": "not_found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{video_id}/annotation")
async def delete_annotation(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete annotation and pose data for a video"""
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    deleted_files = []
    errors = []
    
    # Delete annotated video
    annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
    if annotated_file.exists():
        try:
            annotated_file.unlink()
            deleted_files.append(str(annotated_file))
        except Exception as e:
            errors.append(f"Failed to delete annotated video: {e}")
    
    # Delete T-LEAP pose data
    tleap_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    if tleap_file.exists():
        try:
            tleap_file.unlink()
            deleted_files.append(str(tleap_file))
        except Exception as e:
            errors.append(f"Failed to delete pose data: {e}")
    
    # Delete YOLO detections
    yolo_file = RESULTS_DIR / "yolo" / f"{video_id}_yolo.json"
    if yolo_file.exists():
        try:
            yolo_file.unlink()
            deleted_files.append(str(yolo_file))
        except Exception as e:
            errors.append(f"Failed to delete YOLO data: {e}")
    
    # Clear render status in annotation-renderer
    try:
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{ANNOTATION_RENDERER_URL}/status/{video_id}",
                timeout=5.0
            )
    except:
        pass  # Ignore if service is unavailable
    
    return {
        "video_id": video_id,
        "deleted_files": deleted_files,
        "errors": errors,
        "success": len(errors) == 0
    }


@router.get("/{video_id}/detections")
async def get_video_detections(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get YOLO detections for a video"""
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    yolo_file = RESULTS_DIR / "yolo" / f"{video_id}_yolo.json"
    
    if not yolo_file.exists():
        raise HTTPException(status_code=404, detail="Detections not found")
    
    with open(yolo_file) as f:
        return json.load(f)


@router.get("/{video_id}/pose")
async def get_video_pose(
    video_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get T-LEAP pose data for a video"""
    query = select(Video).where(Video.id == video_id)
    query = _apply_tenant_filter(query, user)
    result = await db.execute(query)
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    pose_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    
    if not pose_file.exists():
        raise HTTPException(status_code=404, detail="Pose data not found")
    
    with open(pose_file) as f:
        return json.load(f)


@router.get("")
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    label_filter: Optional[int] = Query(None, alias="label", description="Filter by label (0=sound, 1=lame)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all videos from database with optional filtering"""
    # Build query
    query = select(Video)
    query = _apply_tenant_filter(query, user)

    # Apply filters
    if label_filter is not None:
        query = query.where(Video.label == label_filter)
    if status:
        query = query.where(Video.status == status)

    # Get total count (with filters applied)
    count_query = select(func.count(Video.id))
    count_query = _apply_tenant_filter(count_query, user)
    if label_filter is not None:
        count_query = count_query.where(Video.label == label_filter)
    if status:
        count_query = count_query.where(Video.status == status)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results, ordered by upload time (newest first)
    query = query.offset(skip).limit(limit).order_by(Video.uploaded_at.desc())
    result = await db.execute(query)
    videos = result.scalars().all()

    return {
        "videos": [
            {
                "video_id": v.id,
                "filename": v.filename,
                "original_filename": v.original_filename,
                "file_size": v.file_size,
                "storage": v.storage_backend,
                "s3_key": v.s3_key,
                "file_path": v.file_path,
                "tenant_id": str(v.tenant_id) if v.tenant_id else None,
                "label": v.label,
                "has_label": v.label is not None,
                "has_analysis": v.has_analysis,
                "has_annotated": v.has_annotated,
                "status": v.status,
                "uploaded_at": v.uploaded_at.isoformat() if v.uploaded_at else None
            }
            for v in videos
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.post("/migrate-to-db")
async def migrate_videos_to_db(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """One-time migration of existing videos to database.

    Scans S3 bucket (if enabled) or local storage and creates database records
    for any videos not already in the database.
    """
    migrated = 0
    skipped = 0
    errors = []

    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    default_tenant = await _get_or_create_default_tenant(db)

    if is_s3_enabled():
        # Scan S3 bucket
        s3 = get_s3_client()
        try:
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=S3_VIDEOS_BUCKET, Prefix='raw/'):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    if not filename:
                        continue
                    video_id = Path(filename).stem

                    # Check if already in DB
                    existing = await db.execute(
                        select(Video).where(Video.id == video_id)
                    )
                    if existing.scalar_one_or_none():
                        skipped += 1
                        continue

                    # Create record
                    try:
                        # Convert timezone-aware datetime to naive UTC
                        last_modified = obj.get('LastModified')
                        if last_modified and last_modified.tzinfo is not None:
                            last_modified = last_modified.replace(tzinfo=None)

                        video = Video(
                            id=video_id,
                            filename=filename,
                            file_size=obj['Size'],
                            storage_backend="s3",
                            s3_key=key,
                            tenant_id=default_tenant.id,
                            status="uploaded",
                            uploaded_at=last_modified or datetime.utcnow()
                        )
                        db.add(video)
                        migrated += 1
                    except Exception as e:
                        errors.append(f"Failed to migrate {video_id}: {str(e)}")

            await db.commit()
        except ClientError as e:
            errors.append(f"S3 error: {str(e)}")
    else:
        # Scan local storage
        if VIDEOS_DIR.exists():
            for video_file in VIDEOS_DIR.glob("*.*"):
                if not video_file.is_file():
                    continue

                video_id = video_file.stem

                # Check if already in DB
                existing = await db.execute(
                    select(Video).where(Video.id == video_id)
                )
                if existing.scalar_one_or_none():
                    skipped += 1
                    continue

                # Create record
                try:
                    video = Video(
                        id=video_id,
                        filename=video_file.name,
                        file_size=video_file.stat().st_size,
                        storage_backend="local",
                        file_path=str(video_file),
                        tenant_id=default_tenant.id,
                        status="uploaded",
                        uploaded_at=datetime.utcnow()
                    )
                    db.add(video)
                    migrated += 1
                except Exception as e:
                    errors.append(f"Failed to migrate {video_id}: {str(e)}")

            await db.commit()

    return {
        "migrated": migrated,
        "skipped": skipped,
        "errors": errors,
        "storage_backend": get_storage_backend()
    }
