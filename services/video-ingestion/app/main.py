"""
Video Ingestion Service
Handles video upload, validation, and storage
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid
from datetime import datetime
import asyncio
import nats
from pathlib import Path
import yaml

# Load config
config_path = Path("/app/shared/config/config.yaml")
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    # Fallback config
    config = {
        "nats": {
            "url": os.getenv("NATS_URL", "nats://localhost:4222"),
            "subjects": {
                "video_uploaded": "video.uploaded"
            }
        }
    }

app = FastAPI(title="Video Ingestion Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Video storage directory
VIDEOS_DIR = Path("/app/data/videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# NATS connection
nc = None


class VideoMetadata(BaseModel):
    video_id: str
    filename: str
    file_path: str
    file_size: int
    uploaded_at: str
    metadata: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize NATS connection"""
    global nc
    nats_url = os.getenv("NATS_URL", config["nats"]["url"])
    try:
        nc = await nats.connect(nats_url)
        print(f"Connected to NATS at {nats_url}")
    except Exception as e:
        print(f"Failed to connect to NATS: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close NATS connection"""
    global nc
    if nc:
        await nc.close()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "video-ingestion"}


@app.post("/upload", response_model=VideoMetadata)
async def upload_video(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """
    Upload a video file
    
    - **file**: Video file (MP4, AVI, MOV)
    - **metadata**: Optional JSON metadata string
    """
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{file_ext}"
    file_path = VIDEOS_DIR / filename
    
    # Save file
    try:
        file_size = 0
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
                file_size += len(chunk)
        
        # Parse metadata if provided
        video_metadata = {}
        if metadata:
            import json
            try:
                video_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                pass
        
        # Create video metadata
        video_info = VideoMetadata(
            video_id=video_id,
            filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            uploaded_at=datetime.utcnow().isoformat(),
            metadata=video_metadata
        )
        
        # Publish to NATS
        if nc:
            import json
            message = {
                "video_id": video_id,
                "file_path": str(file_path),
                "filename": file.filename,
                "file_size": file_size,
                "uploaded_at": video_info.uploaded_at,
                "metadata": video_metadata
            }
            await nc.publish(
                config["nats"]["subjects"]["video_uploaded"],
                json.dumps(message).encode()
            )
            print(f"Published video.uploaded event for {video_id}")
        
        return video_info
        
    except Exception as e:
        # Clean up file if upload failed
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")


@app.get("/videos/{video_id}")
async def get_video_info(video_id: str):
    """Get video information"""
    # Find video file
    video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_path = video_files[0]
    file_size = file_path.stat().st_size
    
    return {
        "video_id": video_id,
        "filename": file_path.name,
        "file_path": str(file_path),
        "file_size": file_size,
        "exists": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

