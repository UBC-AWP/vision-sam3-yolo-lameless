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

from app.routers import videos, analysis, training, models, shap, cows
from app.routers import auth, pipeline, health, ml_config, elo_ranking, tutorial
from app.database import init_db, close_db
from app.websocket.handler import ws_manager, websocket_endpoint

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Arbutus troubleshooting: set BOTO_LOG_LEVEL=DEBUG to log full S3 request/response (incl. headers)
if os.getenv("BOTO_LOG_LEVEL", "").upper() == "DEBUG":
    logging.getLogger("botocore").setLevel(logging.DEBUG)
    logging.getLogger("boto3").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)


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
app.include_router(tutorial.router, prefix="/api/tutorial", tags=["tutorial"])
app.include_router(cows.router, prefix="/api/cows", tags=["cows"])


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


# Database seed endpoint (one-time use for initial setup)
@app.post("/api/seed-db")
async def seed_database():
    """
    Seed the database with tenants Farm1, Farm2 and sample users for demo:
    admin, tenant_admin (per farm), researchers, viewers, rater.
    All demo passwords: demo123
    """
    from app.database import get_db, User, Tenant
    from app.middleware.auth import get_password_hash
    from sqlalchemy import select
    import uuid
    from datetime import datetime

    DEMO_PASSWORD = "demo123"
    created_list = []

    async for db in get_db():
        try:
            # Ensure tenants Farm1 and Farm2 exist (create or rename from farm1/farm2)
            for old_name, new_name in [("farm1", "Farm1"), ("farm2", "Farm2")]:
                r_old = await db.execute(select(Tenant).where(Tenant.name == old_name))
                r_new = await db.execute(select(Tenant).where(Tenant.name == new_name))
                old_row = r_old.scalar_one_or_none()
                new_row = r_new.scalar_one_or_none()
                if new_row:
                    pass
                elif old_row:
                    old_row.name = new_name
                else:
                    t = Tenant(id=uuid.uuid4(), name=new_name, is_active=True)
                    db.add(t)
            await db.flush()

            r1 = await db.execute(select(Tenant).where(Tenant.name == "Farm1"))
            r2 = await db.execute(select(Tenant).where(Tenant.name == "Farm2"))
            farm1 = r1.scalar_one()
            farm2 = r2.scalar_one()

            async def add_user(email: str, username: str, role: str, tenant_id=None, rater_tier=None):
                r = await db.execute(select(User).where(User.email == email))
                if r.scalar_one_or_none():
                    return False
                u = User(
                    id=uuid.uuid4(),
                    email=email,
                    username=username,
                    password_hash=get_password_hash(DEMO_PASSWORD),
                    role=role,
                    is_active=True,
                    rater_tier=rater_tier,
                    tenant_id=tenant_id,
                    created_at=datetime.utcnow(),
                )
                db.add(u)
                created_list.append({"email": email, "username": username, "role": role, "tenant": "Farm1" if tenant_id == farm1.id else ("Farm2" if tenant_id == farm2.id else None)})
                return True

            farm1_id, farm2_id = farm1.id, farm2.id

            # Skip if already seeded (admin exists)
            existing = await db.execute(select(User).where(User.email == "admin@example.com"))
            if existing.scalar_one_or_none():
                await db.commit()
                return {"message": "Database already seeded", "status": "skipped"}

            # System admin (no tenant)
            admin = User(
                id=uuid.UUID("a0000000-0000-0000-0000-000000000001"),
                email="admin@example.com",
                username="admin",
                password_hash=get_password_hash(DEMO_PASSWORD),
                role="admin",
                is_active=True,
                rater_tier="gold",
                tenant_id=None,
                created_at=datetime.utcnow(),
            )
            db.add(admin)
            created_list.append({"email": "admin@example.com", "username": "admin", "role": "admin", "tenant": None})

            # Rater (no tenant, for Pairwise etc.)
            rater = User(
                id=uuid.UUID("a0000000-0000-0000-0000-000000000003"),
                email="rater@example.com",
                username="rater",
                password_hash=get_password_hash(DEMO_PASSWORD),
                role="rater",
                is_active=True,
                rater_tier="bronze",
                tenant_id=None,
                created_at=datetime.utcnow(),
            )
            db.add(rater)
            created_list.append({"email": "rater@example.com", "username": "rater", "role": "rater", "tenant": None})

            # Farm1: tenant_admin, researchers, viewers
            await add_user("tenant_admin_farm1@example.com", "tenant_admin_farm1", "tenant_admin", farm1_id)
            await add_user("researcher_farm1@example.com", "researcher_farm1", "researcher", farm1_id)
            await add_user("researcher_farm1b@example.com", "researcher_farm1b", "researcher", farm1_id)
            await add_user("viewer_farm1@example.com", "viewer_farm1", "viewer", farm1_id)
            await add_user("viewer_farm1b@example.com", "viewer_farm1b", "viewer", farm1_id)

            # Farm2: tenant_admin, researchers, viewers
            await add_user("tenant_admin_farm2@example.com", "tenant_admin_farm2", "tenant_admin", farm2_id)
            await add_user("researcher_farm2@example.com", "researcher_farm2", "researcher", farm2_id)
            await add_user("researcher_farm2b@example.com", "researcher_farm2b", "researcher", farm2_id)
            await add_user("viewer_farm2@example.com", "viewer_farm2", "viewer", farm2_id)
            await add_user("viewer_farm2b@example.com", "viewer_farm2b", "viewer", farm2_id)

            await db.commit()
            return {
                "message": "Database seeded successfully. All demo passwords: " + DEMO_PASSWORD,
                "status": "success",
                "users_created": created_list,
            }
        except Exception as e:
            await db.rollback()
            return {"message": f"Error seeding database: {str(e)}", "status": "error"}


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
