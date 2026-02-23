"""
Cow management API endpoints.
Provides access to cow identities, lameness history, and aggregated predictions.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from app.database import get_db, CowIdentity, TrackHistory, LamenessRecord

router = APIRouter()

RESULTS_DIR = Path("/app/data/results")
COW_PREDICTIONS_DIR = Path("/app/data/results/cow_predictions")
TRACKING_DIR = Path("/app/data/results/tracking")


@router.get("")
async def list_cows(
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    is_active: Optional[bool] = None,
    severity_filter: Optional[str] = None
):
    """
    List all cows with summary statistics.
    
    Args:
        skip: Number of records to skip
        limit: Maximum records to return
        is_active: Filter by active status
        severity_filter: Filter by severity level (healthy, mild, moderate, severe)
    """
    # Build query
    query = select(CowIdentity)
    
    if is_active is not None:
        query = query.where(CowIdentity.is_active == is_active)
    
    query = query.order_by(desc(CowIdentity.last_seen)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    cows = result.scalars().all()
    
    # Get total count
    count_query = select(func.count(CowIdentity.id))
    if is_active is not None:
        count_query = count_query.where(CowIdentity.is_active == is_active)
    count_result = await db.execute(count_query)
    total = count_result.scalar()
    
    # Enrich with latest predictions
    cow_list = []
    for cow in cows:
        cow_data = {
            "id": str(cow.id),
            "cow_id": cow.cow_id,
            "tag_number": cow.tag_number,
            "total_sightings": cow.total_sightings,
            "first_seen": cow.first_seen.isoformat() if cow.first_seen else None,
            "last_seen": cow.last_seen.isoformat() if cow.last_seen else None,
            "is_active": cow.is_active,
            "notes": cow.notes
        }
        
        # Get latest prediction from file
        prediction_file = COW_PREDICTIONS_DIR / f"{cow.cow_id}_prediction.json"
        if prediction_file.exists():
            try:
                with open(prediction_file) as f:
                    pred_data = json.load(f)
                prediction = pred_data.get("prediction", {})
                cow_data["current_score"] = prediction.get("aggregated_score")
                cow_data["severity_level"] = prediction.get("severity_level")
                cow_data["num_videos"] = prediction.get("num_videos", 0)
            except:
                cow_data["current_score"] = None
                cow_data["severity_level"] = None
                cow_data["num_videos"] = 0
        else:
            cow_data["current_score"] = None
            cow_data["severity_level"] = None
            cow_data["num_videos"] = 0
        
        # Filter by severity if requested
        if severity_filter:
            if cow_data.get("severity_level") != severity_filter:
                continue
        
        cow_list.append(cow_data)
    
    return {
        "cows": cow_list,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get("/{cow_id}")
async def get_cow(cow_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get detailed information about a specific cow.
    """
    # Try to find by cow_id string
    result = await db.execute(
        select(CowIdentity).where(CowIdentity.cow_id == cow_id)
    )
    cow = result.scalar_one_or_none()
    
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    # Get video count
    video_result = await db.execute(
        select(func.count(TrackHistory.id)).where(TrackHistory.cow_id == cow.id)
    )
    video_count = video_result.scalar() or 0
    
    # Get lameness record count
    record_result = await db.execute(
        select(func.count(LamenessRecord.id)).where(LamenessRecord.cow_id == cow.id)
    )
    record_count = record_result.scalar() or 0
    
    # Get latest prediction
    prediction_data = None
    prediction_file = COW_PREDICTIONS_DIR / f"{cow.cow_id}_prediction.json"
    if prediction_file.exists():
        try:
            with open(prediction_file) as f:
                prediction_data = json.load(f)
        except:
            pass
    
    return {
        "id": str(cow.id),
        "cow_id": cow.cow_id,
        "tag_number": cow.tag_number,
        "total_sightings": cow.total_sightings,
        "first_seen": cow.first_seen.isoformat() if cow.first_seen else None,
        "last_seen": cow.last_seen.isoformat() if cow.last_seen else None,
        "is_active": cow.is_active,
        "notes": cow.notes,
        "embedding_version": cow.embedding_version,
        "video_count": video_count,
        "lameness_record_count": record_count,
        "current_prediction": prediction_data.get("prediction") if prediction_data else None,
        "last_prediction_update": prediction_data.get("last_updated") if prediction_data else None
    }


@router.get("/{cow_id}/lameness")
async def get_cow_lameness_history(
    cow_id: str,
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """
    Get lameness history timeline for a cow.
    
    Args:
        cow_id: The cow identifier
        days: Number of days of history to return
    """
    # Find cow
    result = await db.execute(
        select(CowIdentity).where(CowIdentity.cow_id == cow_id)
    )
    cow = result.scalar_one_or_none()
    
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    # Get lameness records
    since = datetime.utcnow() - timedelta(days=days)
    records_result = await db.execute(
        select(LamenessRecord)
        .where(LamenessRecord.cow_id == cow.id)
        .where(LamenessRecord.observation_date >= since)
        .order_by(desc(LamenessRecord.observation_date))
    )
    records = records_result.scalars().all()
    
    # Format for timeline
    timeline = []
    for record in records:
        timeline.append({
            "id": str(record.id),
            "video_id": record.video_id,
            "date": record.observation_date.isoformat() if record.observation_date else None,
            "fusion_score": record.fusion_score,
            "is_lame": record.is_lame,
            "confidence": record.confidence,
            "severity_level": record.severity_level,
            "pipeline_scores": {
                "tleap": record.tleap_score,
                "tcn": record.tcn_score,
                "transformer": record.transformer_score,
                "gnn": record.gnn_score,
                "graph_transformer": record.graph_transformer_score,
                "ml_ensemble": record.ml_ensemble_score
            },
            "human_validated": record.human_validated,
            "human_label": record.human_label
        })
    
    # Compute trend
    if len(timeline) >= 2:
        recent_scores = [r["fusion_score"] for r in timeline[:5] if r["fusion_score"] is not None]
        older_scores = [r["fusion_score"] for r in timeline[5:10] if r["fusion_score"] is not None]
        
        if recent_scores and older_scores:
            trend = sum(recent_scores) / len(recent_scores) - sum(older_scores) / len(older_scores)
            if trend > 0.1:
                trend_direction = "worsening"
            elif trend < -0.1:
                trend_direction = "improving"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "unknown"
    else:
        trend_direction = "insufficient_data"
    
    return {
        "cow_id": cow_id,
        "timeline": timeline,
        "total_records": len(records),
        "days_range": days,
        "trend": trend_direction
    }


@router.get("/{cow_id}/videos")
async def get_cow_videos(
    cow_id: str,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get all videos for a specific cow with their analysis results.
    """
    # Find cow
    result = await db.execute(
        select(CowIdentity).where(CowIdentity.cow_id == cow_id)
    )
    cow = result.scalar_one_or_none()
    
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    # Get track history (videos)
    tracks_result = await db.execute(
        select(TrackHistory)
        .where(TrackHistory.cow_id == cow.id)
        .order_by(desc(TrackHistory.created_at))
        .offset(skip)
        .limit(limit)
    )
    tracks = tracks_result.scalars().all()
    
    # Get total count
    count_result = await db.execute(
        select(func.count(TrackHistory.id)).where(TrackHistory.cow_id == cow.id)
    )
    total = count_result.scalar()
    
    # Enrich with fusion results
    videos = []
    for track in tracks:
        video_data = {
            "video_id": track.video_id,
            "track_id": track.track_id,
            "reid_confidence": track.reid_confidence,
            "start_frame": track.start_frame,
            "end_frame": track.end_frame,
            "total_frames": track.total_frames,
            "created_at": track.created_at.isoformat() if track.created_at else None
        }
        
        # Load fusion result
        fusion_file = RESULTS_DIR / "fusion" / f"{track.video_id}_fusion.json"
        if fusion_file.exists():
            try:
                with open(fusion_file) as f:
                    fusion_data = json.load(f)
                fusion_result = fusion_data.get("fusion_result", {})
                video_data["lameness_score"] = fusion_result.get("final_probability")
                video_data["prediction"] = fusion_result.get("final_prediction")
                video_data["confidence"] = fusion_result.get("confidence")
            except:
                video_data["lameness_score"] = None
        else:
            video_data["lameness_score"] = None
        
        videos.append(video_data)
    
    return {
        "cow_id": cow_id,
        "videos": videos,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get("/{cow_id}/prediction")
async def get_cow_prediction(cow_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get the current aggregated lameness prediction for a cow.
    """
    # Verify cow exists
    result = await db.execute(
        select(CowIdentity).where(CowIdentity.cow_id == cow_id)
    )
    cow = result.scalar_one_or_none()
    
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    # Load prediction from file
    prediction_file = COW_PREDICTIONS_DIR / f"{cow_id}_prediction.json"
    if not prediction_file.exists():
        raise HTTPException(status_code=404, detail="No prediction available for this cow")
    
    try:
        with open(prediction_file) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading prediction: {e}")
    
    return {
        "cow_id": cow_id,
        "prediction": data.get("prediction", {}),
        "last_updated": data.get("last_updated"),
        "latest_video": data.get("latest_video")
    }


@router.patch("/{cow_id}")
async def update_cow(
    cow_id: str,
    tag_number: Optional[str] = None,
    notes: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Update cow information.
    """
    result = await db.execute(
        select(CowIdentity).where(CowIdentity.cow_id == cow_id)
    )
    cow = result.scalar_one_or_none()
    
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    if tag_number is not None:
        cow.tag_number = tag_number
    if notes is not None:
        cow.notes = notes
    if is_active is not None:
        cow.is_active = is_active
    
    await db.commit()
    
    return {
        "id": str(cow.id),
        "cow_id": cow.cow_id,
        "tag_number": cow.tag_number,
        "notes": cow.notes,
        "is_active": cow.is_active,
        "message": "Cow updated successfully"
    }


@router.get("/{cow_id}/lameness/{record_id}/validate")
async def validate_lameness_record(
    cow_id: str,
    record_id: str,
    is_lame: bool,
    validator_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Validate a lameness record with human label.
    """
    # Find the record
    from uuid import UUID
    try:
        record_uuid = UUID(record_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid record ID format")
    
    result = await db.execute(
        select(LamenessRecord).where(LamenessRecord.id == record_uuid)
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="Lameness record not found")
    
    # Update record
    record.human_validated = True
    record.human_label = is_lame
    record.validation_date = datetime.utcnow()
    
    if validator_id:
        try:
            record.validator_id = UUID(validator_id)
        except:
            pass
    
    await db.commit()
    
    return {
        "record_id": record_id,
        "human_validated": True,
        "human_label": is_lame,
        "message": "Record validated successfully"
    }


@router.get("/stats/summary")
async def get_cow_stats_summary(db: AsyncSession = Depends(get_db)):
    """
    Get summary statistics for all cows.
    """
    # Total cows
    total_result = await db.execute(select(func.count(CowIdentity.id)))
    total_cows = total_result.scalar() or 0
    
    # Active cows
    active_result = await db.execute(
        select(func.count(CowIdentity.id)).where(CowIdentity.is_active == True)
    )
    active_cows = active_result.scalar() or 0
    
    # Total videos tracked
    videos_result = await db.execute(select(func.count(TrackHistory.id)))
    total_videos = videos_result.scalar() or 0
    
    # Total lameness records
    records_result = await db.execute(select(func.count(LamenessRecord.id)))
    total_records = records_result.scalar() or 0
    
    # Severity distribution from files
    severity_counts = {"healthy": 0, "mild": 0, "moderate": 0, "severe": 0, "unknown": 0}
    
    if COW_PREDICTIONS_DIR.exists():
        for pred_file in COW_PREDICTIONS_DIR.glob("*_prediction.json"):
            try:
                with open(pred_file) as f:
                    data = json.load(f)
                severity = data.get("prediction", {}).get("severity_level", "unknown")
                if severity in severity_counts:
                    severity_counts[severity] += 1
                else:
                    severity_counts["unknown"] += 1
            except:
                severity_counts["unknown"] += 1
    
    return {
        "total_cows": total_cows,
        "active_cows": active_cows,
        "total_videos_tracked": total_videos,
        "total_lameness_records": total_records,
        "severity_distribution": severity_counts
    }

