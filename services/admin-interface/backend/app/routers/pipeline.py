"""
Pipeline monitoring and control endpoints
Provides status, logs, and manual triggering for all ML pipelines
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import nats

from app.database import get_db, ServiceHeartbeat, User
from app.middleware.auth import get_current_user, require_role
from app.websocket.handler import ws_manager

router = APIRouter()

# Pipeline service definitions
PIPELINE_SERVICES = {
    "video-preprocessing": {"description": "Video preprocessing and frame extraction", "port": None},
    "clip-curation": {"description": "Clip curation and selection", "port": None},
    "yolo-pipeline": {"description": "YOLO object detection for cow detection", "port": None},
    "sam3-pipeline": {"description": "SAM3 segmentation for cow masking", "port": None},
    "dinov3-pipeline": {"description": "DINOv3 feature extraction for embeddings", "port": None},
    "tleap-pipeline": {"description": "T-LEAP pose estimation for cow skeleton", "port": None},
    "tcn-pipeline": {"description": "TCN temporal analysis for gait patterns", "port": None},
    "transformer-pipeline": {"description": "Transformer-based lameness prediction", "port": None},
    "gnn-pipeline": {"description": "Graph Neural Network for structural analysis", "port": None},
    "fusion-service": {"description": "Multi-pipeline result fusion", "port": None},
}

# Data directories
RESULTS_DIR = Path("/app/data/results")
LOGS_DIR = Path("/app/data/logs")

# NATS connection
_nats_client = None


async def get_nats():
    """Get NATS connection"""
    global _nats_client
    if _nats_client is None or not _nats_client.is_connected:
        nats_url = os.getenv("NATS_URL", "nats://nats:4222")
        _nats_client = await nats.connect(nats_url)
    return _nats_client


# ============== MODELS ==============

class PipelineStatus(BaseModel):
    """Pipeline status response"""
    service_name: str
    description: str
    status: str  # healthy, degraded, down, unknown
    last_heartbeat: Optional[datetime] = None
    active_jobs: int = 0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    last_error: Optional[str] = None


class PipelineTriggerRequest(BaseModel):
    """Request to trigger a pipeline"""
    video_id: str
    options: Optional[Dict[str, Any]] = None


class BatchReprocessRequest(BaseModel):
    """Request to batch reprocess videos"""
    video_ids: List[str]
    pipelines: Optional[List[str]] = None  # None means all pipelines


class LogEntry(BaseModel):
    """Log entry model"""
    timestamp: datetime
    level: str
    service: str
    message: str


# ============== ENDPOINTS ==============

@router.get("/status", response_model=List[PipelineStatus])
async def get_all_pipeline_status(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get status of all pipeline services.
    """
    statuses = []
    now = datetime.utcnow()
    heartbeat_timeout = timedelta(seconds=30)

    for service_name, service_info in PIPELINE_SERVICES.items():
        # Check database for heartbeat
        result = await db.execute(
            select(ServiceHeartbeat).where(ServiceHeartbeat.service_name == service_name)
        )
        heartbeat = result.scalar_one_or_none()

        if heartbeat and heartbeat.last_heartbeat:
            time_since_heartbeat = now - heartbeat.last_heartbeat
            if time_since_heartbeat < heartbeat_timeout:
                status = heartbeat.status or "healthy"
            elif time_since_heartbeat < heartbeat_timeout * 2:
                status = "degraded"
            else:
                status = "down"

            total = heartbeat.success_count + heartbeat.error_count
            success_rate = heartbeat.success_count / total if total > 0 else 0.0

            statuses.append(PipelineStatus(
                service_name=service_name,
                description=service_info["description"],
                status=status,
                last_heartbeat=heartbeat.last_heartbeat,
                active_jobs=heartbeat.active_jobs,
                success_count=heartbeat.success_count,
                error_count=heartbeat.error_count,
                success_rate=success_rate,
                last_error=heartbeat.last_error
            ))
        else:
            statuses.append(PipelineStatus(
                service_name=service_name,
                description=service_info["description"],
                status="unknown",
                active_jobs=0
            ))

    return statuses


@router.get("/{service_name}/status", response_model=PipelineStatus)
async def get_pipeline_status(
    service_name: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get status of a specific pipeline service.
    """
    if service_name not in PIPELINE_SERVICES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_name}")

    service_info = PIPELINE_SERVICES[service_name]
    now = datetime.utcnow()
    heartbeat_timeout = timedelta(seconds=30)

    result = await db.execute(
        select(ServiceHeartbeat).where(ServiceHeartbeat.service_name == service_name)
    )
    heartbeat = result.scalar_one_or_none()

    if heartbeat and heartbeat.last_heartbeat:
        time_since_heartbeat = now - heartbeat.last_heartbeat
        if time_since_heartbeat < heartbeat_timeout:
            status = heartbeat.status or "healthy"
        elif time_since_heartbeat < heartbeat_timeout * 2:
            status = "degraded"
        else:
            status = "down"

        total = heartbeat.success_count + heartbeat.error_count
        success_rate = heartbeat.success_count / total if total > 0 else 0.0

        return PipelineStatus(
            service_name=service_name,
            description=service_info["description"],
            status=status,
            last_heartbeat=heartbeat.last_heartbeat,
            active_jobs=heartbeat.active_jobs,
            success_count=heartbeat.success_count,
            error_count=heartbeat.error_count,
            success_rate=success_rate,
            last_error=heartbeat.last_error
        )

    return PipelineStatus(
        service_name=service_name,
        description=service_info["description"],
        status="unknown",
        active_jobs=0
    )


@router.get("/{service_name}/logs", response_model=List[LogEntry])
async def get_pipeline_logs(
    service_name: str,
    limit: int = Query(100, ge=1, le=1000),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR)"),
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get recent logs for a pipeline service.
    """
    if service_name not in PIPELINE_SERVICES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_name}")

    logs = []
    log_file = LOGS_DIR / f"{service_name}.log"

    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()[-limit * 2:]  # Read extra to account for filtering

            for line in reversed(lines):
                try:
                    # Parse log line (assuming JSON format)
                    entry = json.loads(line.strip())
                    log_entry = LogEntry(
                        timestamp=datetime.fromisoformat(entry.get("timestamp", datetime.utcnow().isoformat())),
                        level=entry.get("level", "INFO"),
                        service=service_name,
                        message=entry.get("message", line.strip())
                    )

                    if level is None or log_entry.level == level:
                        logs.append(log_entry)

                    if len(logs) >= limit:
                        break
                except json.JSONDecodeError:
                    # Handle non-JSON log lines
                    log_entry = LogEntry(
                        timestamp=datetime.utcnow(),
                        level="INFO",
                        service=service_name,
                        message=line.strip()
                    )
                    if level is None or level == "INFO":
                        logs.append(log_entry)

                    if len(logs) >= limit:
                        break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

    return logs


@router.post("/{service_name}/trigger/{video_id}")
async def trigger_pipeline(
    service_name: str,
    video_id: str,
    request: Optional[PipelineTriggerRequest] = None,
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Manually trigger a pipeline for a specific video.
    """
    if service_name not in PIPELINE_SERVICES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_name}")

    # Normalize video_id if user includes extension
    raw_video_id = video_id
    suffix = Path(video_id).suffix.lower()
    allowed_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    file_path = None
    if suffix in allowed_suffixes:
        video_id = video_id[: -len(suffix)]
        file_path = f"/app/data/videos/{video_id}{suffix}"

    # Map service to NATS subject
    subject_map = {
        "video-preprocessing": "video.uploaded",
        "clip-curation": "video.preprocessed",
        "yolo-pipeline": "video.preprocessed",
        "sam3-pipeline": "video.preprocessed",
        "dinov3-pipeline": "video.preprocessed",
        "tleap-pipeline": "video.preprocessed",
        "tcn-pipeline": "pipeline.tleap",
        "transformer-pipeline": "pipeline.tleap",
        "gnn-pipeline": "pipeline.dinov3",
        "fusion-service": "pipeline.ml",
    }

    subject = subject_map.get(service_name)
    if not subject:
        raise HTTPException(status_code=400, detail=f"Cannot trigger {service_name} manually")

    try:
        nc = await get_nats()
        msg = {
            "video_id": video_id,
            "triggered_by": str(user.id),
            "triggered_at": datetime.utcnow().isoformat(),
            "options": request.options if request else {}
        }
        if file_path:
            msg["file_path"] = file_path
            msg["filename"] = Path(file_path).name
        await nc.publish(subject, json.dumps(msg).encode())
        await nc.flush()

        # Broadcast update via WebSocket
        await ws_manager.broadcast_pipeline_status(
            service_name,
            "triggered",
            {"video_id": video_id, "triggered_by": user.username}
        )

        return {
            "status": "triggered",
            "service": service_name,
            "video_id": video_id,
            "input_video_id": raw_video_id,
            "subject": subject
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger pipeline: {str(e)}")


@router.post("/batch/reprocess")
async def batch_reprocess(
    request: BatchReprocessRequest,
    user: User = Depends(require_role(["admin"]))
):
    """
    Batch reprocess multiple videos through specified pipelines.
    Admin only.
    """
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="No video IDs provided")

    pipelines = request.pipelines or list(PIPELINE_SERVICES.keys())
    triggered = []
    errors = []

    try:
        nc = await get_nats()

        for video_id in request.video_ids:
            for pipeline in pipelines:
                if pipeline not in PIPELINE_SERVICES:
                    errors.append({"video_id": video_id, "pipeline": pipeline, "error": "Unknown pipeline"})
                    continue

                subject_map = {
                    "video-preprocessing": "video.uploaded",
                    "yolo-pipeline": "video.preprocessed",
                    "sam3-pipeline": "video.preprocessed",
                    "dinov3-pipeline": "video.preprocessed",
                    "tleap-pipeline": "video.preprocessed",
                }

                subject = subject_map.get(pipeline)
                if subject:
                    # Include processed_path for pipelines that need it
                    video_path = f"/app/data/videos/{video_id}.mp4"
                    msg = {
                        "video_id": video_id,
                        "processed_path": video_path,
                        "triggered_by": str(user.id),
                        "batch_reprocess": True
                    }
                    await nc.publish(subject, json.dumps(msg).encode())
                    triggered.append({"video_id": video_id, "pipeline": pipeline})

        await nc.flush()

        return {
            "status": "batch_triggered",
            "triggered_count": len(triggered),
            "error_count": len(errors),
            "triggered": triggered,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch reprocess failed: {str(e)}")


@router.get("/queue")
async def get_queue_depths(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get NATS queue depths for monitoring.
    """
    try:
        nc = await get_nats()
        js = nc.jetstream()

        # Stream names to check
        streams = [
            "video-stream",
            "pipeline-stream",
            "results-stream"
        ]

        queue_info = {}
        for stream_name in streams:
            try:
                stream = await js.stream_info(stream_name)
                queue_info[stream_name] = {
                    "messages": stream.state.messages,
                    "bytes": stream.state.bytes,
                    "first_seq": stream.state.first_seq,
                    "last_seq": stream.state.last_seq,
                    "consumer_count": stream.state.consumer_count
                }
            except Exception:
                queue_info[stream_name] = {"status": "not_found"}

        return queue_info
    except Exception as e:
        return {"error": str(e), "status": "unavailable"}


@router.post("/heartbeat")
async def record_heartbeat(
    service_name: str,
    status: str = "healthy",
    active_jobs: int = 0,
    success_count: int = 0,
    error_count: int = 0,
    last_error: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Record a heartbeat from a pipeline service.
    Called by pipeline services to report their status.
    """
    if service_name not in PIPELINE_SERVICES:
        raise HTTPException(status_code=400, detail=f"Unknown service: {service_name}")

    result = await db.execute(
        select(ServiceHeartbeat).where(ServiceHeartbeat.service_name == service_name)
    )
    heartbeat = result.scalar_one_or_none()

    if heartbeat:
        heartbeat.status = status
        heartbeat.last_heartbeat = datetime.utcnow()
        heartbeat.active_jobs = active_jobs
        heartbeat.success_count = success_count
        heartbeat.error_count = error_count
        if last_error:
            heartbeat.last_error = last_error
    else:
        import uuid
        heartbeat = ServiceHeartbeat(
            id=uuid.uuid4(),
            service_name=service_name,
            status=status,
            last_heartbeat=datetime.utcnow(),
            active_jobs=active_jobs,
            success_count=success_count,
            error_count=error_count,
            last_error=last_error
        )
        db.add(heartbeat)

    await db.commit()

    # Broadcast status update via WebSocket
    await ws_manager.broadcast_pipeline_status(
        service_name,
        status,
        {"active_jobs": active_jobs}
    )

    return {"status": "recorded", "service": service_name}
