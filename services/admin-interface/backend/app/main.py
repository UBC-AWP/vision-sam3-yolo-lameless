"""
Admin Interface Backend
FastAPI backend for admin interface with authentication and real-time updates
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
import os
import traceback
import logging

from app.routers import videos, analysis, training, models, shap
from app.routers import auth, pipeline, health, ml_config, elo_ranking
from app.database import init_db, close_db
from app.websocket.handler import ws_manager, websocket_endpoint

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events"""
    # Startup: Initialize database tables
    await init_db()
    print("Database initialized")
    yield
    # Shutdown: Close database connections
    await close_db()
    print("Database connections closed")


app = FastAPI(
    title="Lameness Detection Admin API",
    description="Admin interface API for cow lameness detection system with authentication and real-time updates",
    version="2.0.0",
    lifespan=lifespan
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions"""
    error_detail = traceback.format_exc()
    logger.error(f"Unhandled exception: {exc}\n{error_detail}")
    print(f"ERROR: {exc}\n{error_detail}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": error_detail}
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(shap.router, prefix="/api/shap", tags=["shap"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(ml_config.router, prefix="/api/ml-config", tags=["ml-config"])
app.include_router(elo_ranking.router, prefix="/api/elo", tags=["elo-ranking"])


# ============== WEBSOCKET ENDPOINTS ==============

@app.websocket("/api/ws/pipeline")
async def ws_pipeline(websocket: WebSocket):
    """WebSocket endpoint for pipeline status updates"""
    await websocket_endpoint(websocket, "pipeline")


@app.websocket("/api/ws/health")
async def ws_health(websocket: WebSocket):
    """WebSocket endpoint for system health updates"""
    await websocket_endpoint(websocket, "health")


@app.websocket("/api/ws/queue")
async def ws_queue(websocket: WebSocket):
    """WebSocket endpoint for processing queue updates"""
    await websocket_endpoint(websocket, "queue")


@app.websocket("/api/ws/rater")
async def ws_rater(websocket: WebSocket):
    """WebSocket endpoint for rater activity updates"""
    await websocket_endpoint(websocket, "rater")


# Health check
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "admin-backend",
        "websocket_connections": ws_manager.get_connection_count()
    }


# Root
@app.get("/")
async def root():
    return {
        "message": "Lameness Detection Admin API",
        "docs": "/docs",
        "version": "2.0.0",
        "features": [
            "Authentication with JWT",
            "Role-based access control (RBAC)",
            "WebSocket real-time updates",
            "Pipeline monitoring",
            "Processing queue management"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
