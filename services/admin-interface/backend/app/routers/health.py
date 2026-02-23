"""
System health monitoring endpoints
Provides health status for Docker containers, NATS, databases, and disk usage
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import os
import asyncio
import httpx
import psutil
from sqlalchemy import select

from app.database import get_db, User
from app.middleware.auth import get_current_user, require_role

router = APIRouter()

# Service URLs
NATS_MONITORING_URL = os.getenv("NATS_MONITORING_URL", "http://nats:8222")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Data directories
DATA_DIR = Path("/app/data")


# ============== MODELS ==============

class HealthOverview(BaseModel):
    """Overall system health"""
    status: str  # healthy, degraded, critical
    timestamp: datetime
    components: Dict[str, str]
    issues: List[str]


class ContainerHealth(BaseModel):
    """Docker container health info"""
    name: str
    status: str
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    memory_percent: Optional[float] = None
    uptime: Optional[str] = None


class NATSHealth(BaseModel):
    """NATS server health"""
    status: str
    connections: int = 0
    subscriptions: int = 0
    messages_in: int = 0
    messages_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


class DatabaseHealth(BaseModel):
    """Database health info"""
    status: str
    connection_count: int = 0
    database_size_mb: float = 0.0
    response_time_ms: float = 0.0


class DiskUsage(BaseModel):
    """Disk usage info"""
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float
    status: str  # healthy, warning, critical


class ThroughputMetrics(BaseModel):
    """Processing throughput metrics"""
    videos_processed_24h: int = 0
    videos_processed_7d: int = 0
    avg_processing_time_s: float = 0.0
    success_rate: float = 0.0
    queue_depth: int = 0


# ============== ENDPOINTS ==============

@router.get("/overview", response_model=HealthOverview)
async def get_health_overview(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get overall system health status.
    """
    components = {}
    issues = []

    # Check NATS
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{NATS_MONITORING_URL}/varz")
            if response.status_code == 200:
                components["nats"] = "healthy"
            else:
                components["nats"] = "degraded"
                issues.append("NATS monitoring endpoint returned non-200 status")
    except Exception as e:
        components["nats"] = "down"
        issues.append(f"NATS connection failed: {str(e)}")

    # Check Qdrant
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QDRANT_URL}/health")
            if response.status_code == 200:
                components["qdrant"] = "healthy"
            else:
                components["qdrant"] = "degraded"
    except Exception:
        components["qdrant"] = "down"
        issues.append("Qdrant vector database unavailable")

    # Check disk space
    try:
        disk = psutil.disk_usage("/app/data")
        if disk.percent > 90:
            components["disk"] = "critical"
            issues.append(f"Disk usage critical: {disk.percent:.1f}%")
        elif disk.percent > 75:
            components["disk"] = "warning"
            issues.append(f"Disk usage high: {disk.percent:.1f}%")
        else:
            components["disk"] = "healthy"
    except Exception:
        components["disk"] = "unknown"

    # Check data directories exist
    for dir_name in ["videos", "processed", "results", "training"]:
        dir_path = DATA_DIR / dir_name
        if not dir_path.exists():
            issues.append(f"Data directory missing: {dir_name}")

    # Determine overall status
    if "down" in components.values() or "critical" in components.values():
        overall_status = "critical"
    elif "degraded" in components.values() or "warning" in components.values():
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthOverview(
        status=overall_status,
        timestamp=datetime.utcnow(),
        components=components,
        issues=issues
    )


@router.get("/docker", response_model=List[ContainerHealth])
async def get_docker_health(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get health status of Docker containers.
    Note: This endpoint provides simulated data as direct Docker access
    is not available from within a container.
    """
    # In a real deployment, this would query Docker API or use a monitoring agent
    # For now, return known services with estimated status

    services = [
        "nats", "postgres", "qdrant",
        "video-ingestion", "video-preprocessing", "clip-curation",
        "yolo-pipeline", "sam3-pipeline", "dinov3-pipeline",
        "tleap-pipeline", "tcn-pipeline", "transformer-pipeline",
        "gnn-pipeline", "ml-pipeline", "fusion-service",
        "annotation-renderer", "training-service",
        "admin-backend", "admin-frontend"
    ]

    containers = []
    for service in services:
        containers.append(ContainerHealth(
            name=service,
            status="running",
            cpu_percent=None,
            memory_mb=None,
            memory_percent=None,
            uptime=None
        ))

    return containers


@router.get("/nats", response_model=NATSHealth)
async def get_nats_health(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get NATS server health and metrics.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{NATS_MONITORING_URL}/varz")

            if response.status_code != 200:
                return NATSHealth(status="degraded")

            data = response.json()

            return NATSHealth(
                status="healthy",
                connections=data.get("connections", 0),
                subscriptions=data.get("subscriptions", 0),
                messages_in=data.get("in_msgs", 0),
                messages_out=data.get("out_msgs", 0),
                bytes_in=data.get("in_bytes", 0),
                bytes_out=data.get("out_bytes", 0)
            )
    except Exception as e:
        return NATSHealth(status="down")


@router.get("/postgres", response_model=DatabaseHealth)
async def get_postgres_health(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get PostgreSQL database health.
    """
    from sqlalchemy import text
    from app.database import async_session
    import time

    try:
        start = time.time()
        async with async_session() as session:
            # Test connection with simple query
            result = await session.execute(text("SELECT 1"))
            result.scalar()

            # Get connection count
            result = await session.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
            )
            conn_count = result.scalar() or 0

            # Get database size
            result = await session.execute(
                text("SELECT pg_database_size(current_database()) / 1024 / 1024 as size_mb")
            )
            db_size = result.scalar() or 0

        response_time = (time.time() - start) * 1000

        return DatabaseHealth(
            status="healthy",
            connection_count=conn_count,
            database_size_mb=float(db_size),
            response_time_ms=response_time
        )
    except Exception as e:
        return DatabaseHealth(
            status="down",
            response_time_ms=0
        )


@router.get("/qdrant", response_model=DatabaseHealth)
async def get_qdrant_health(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get Qdrant vector database health.
    """
    import time

    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health check
            response = await client.get(f"{QDRANT_URL}/health")

            if response.status_code != 200:
                return DatabaseHealth(status="degraded")

            # Get collections info
            collections_response = await client.get(f"{QDRANT_URL}/collections")
            collections_data = collections_response.json()

            response_time = (time.time() - start) * 1000

            # Calculate approximate size
            total_points = 0
            if "result" in collections_data and "collections" in collections_data["result"]:
                for collection in collections_data["result"]["collections"]:
                    try:
                        coll_response = await client.get(f"{QDRANT_URL}/collections/{collection['name']}")
                        coll_data = coll_response.json()
                        total_points += coll_data.get("result", {}).get("points_count", 0)
                    except Exception:
                        pass

            return DatabaseHealth(
                status="healthy",
                connection_count=1,
                database_size_mb=total_points * 0.004,  # Rough estimate: 4KB per point
                response_time_ms=response_time
            )
    except Exception:
        return DatabaseHealth(status="down")


@router.get("/disk", response_model=List[DiskUsage])
async def get_disk_usage(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get disk usage for data directories.
    """
    paths_to_check = [
        "/app/data",
        "/app/data/videos",
        "/app/data/processed",
        "/app/data/results",
        "/app/data/training"
    ]

    results = []
    for path in paths_to_check:
        try:
            if Path(path).exists():
                usage = psutil.disk_usage(path)
                percent = usage.percent

                if percent >= 90:
                    status = "critical"
                elif percent >= 75:
                    status = "warning"
                else:
                    status = "healthy"

                results.append(DiskUsage(
                    path=path,
                    total_gb=usage.total / (1024**3),
                    used_gb=usage.used / (1024**3),
                    free_gb=usage.free / (1024**3),
                    percent_used=percent,
                    status=status
                ))
            else:
                results.append(DiskUsage(
                    path=path,
                    total_gb=0,
                    used_gb=0,
                    free_gb=0,
                    percent_used=0,
                    status="missing"
                ))
        except Exception:
            results.append(DiskUsage(
                path=path,
                total_gb=0,
                used_gb=0,
                free_gb=0,
                percent_used=0,
                status="error"
            ))

    return results


@router.get("/throughput", response_model=ThroughputMetrics)
async def get_throughput_metrics(
    user: User = Depends(require_role(["admin", "researcher"]))
):
    """
    Get processing throughput metrics.
    """
    from app.database import async_session, ProcessingJob
    from sqlalchemy import func

    try:
        async with async_session() as session:
            now = datetime.utcnow()
            day_ago = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)

            # Videos processed in last 24h
            result = await session.execute(
                select(func.count(ProcessingJob.job_id)).where(
                    ProcessingJob.completed_at >= day_ago,
                    ProcessingJob.status == "completed"
                )
            )
            videos_24h = result.scalar() or 0

            # Videos processed in last 7 days
            result = await session.execute(
                select(func.count(ProcessingJob.job_id)).where(
                    ProcessingJob.completed_at >= week_ago,
                    ProcessingJob.status == "completed"
                )
            )
            videos_7d = result.scalar() or 0

            # Success rate
            result = await session.execute(
                select(func.count(ProcessingJob.job_id)).where(
                    ProcessingJob.created_at >= week_ago
                )
            )
            total = result.scalar() or 0

            success_rate = videos_7d / total if total > 0 else 1.0

            # Queue depth (pending jobs)
            result = await session.execute(
                select(func.count(ProcessingJob.job_id)).where(
                    ProcessingJob.status == "pending"
                )
            )
            queue_depth = result.scalar() or 0

            return ThroughputMetrics(
                videos_processed_24h=videos_24h,
                videos_processed_7d=videos_7d,
                avg_processing_time_s=0,  # Would need additional tracking
                success_rate=success_rate,
                queue_depth=queue_depth
            )
    except Exception:
        return ThroughputMetrics()
