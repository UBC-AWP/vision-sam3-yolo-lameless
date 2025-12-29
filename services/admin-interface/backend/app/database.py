"""
Database connection and session management
Async SQLAlchemy setup for PostgreSQL
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Boolean, DateTime, Float, Integer, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

# Database URL from environment
_raw_url = os.getenv(
    "POSTGRES_URL",
    os.getenv("DATABASE_URL", "postgresql://lameness_user:lameness_pass@postgres:5432/lameness_db")
)
# Convert to asyncpg URL
DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


# ============== MODELS ==============

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="rater")
    is_active = Column(Boolean, default=True)
    rater_tier = Column(String(10), nullable=True)  # gold, silver, bronze
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'researcher', 'rater')", name="valid_role"),
    )


class Session(Base):
    """Session model for token management"""
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessingJob(Base):
    """Processing job model for queue management"""
    __tablename__ = "processing_jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(100), nullable=False, index=True)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed, cancelled
    priority = Column(Integer, default=0)
    progress = Column(Float, default=0.0)
    current_pipeline = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')",
            name="valid_job_status"
        ),
    )


class GoldTask(Base):
    """Gold task model for rater validation and tutorial"""
    __tablename__ = "gold_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id_1 = Column(String(100), nullable=False)
    video_id_2 = Column(String(100), nullable=False)
    correct_winner = Column(Integer, nullable=False)  # 0=tie, 1=video1, 2=video2
    correct_degree = Column(Integer, default=2)  # 1-3 strength of correct answer
    difficulty = Column(String(10), default="medium")  # easy, medium, hard
    description = Column(Text, nullable=True)  # What to look for
    hint = Column(Text, nullable=True)  # Explanation of correct answer
    is_tutorial = Column(Boolean, default=False)  # If true, shown in tutorial mode
    tutorial_order = Column(Integer, nullable=True)  # Order in tutorial sequence
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("correct_winner IN (0, 1, 2)", name="valid_winner"),
        CheckConstraint("difficulty IN ('easy', 'medium', 'hard')", name="valid_difficulty"),
        CheckConstraint("correct_degree >= 1 AND correct_degree <= 3", name="valid_degree"),
    )


class RaterStats(Base):
    """Rater statistics model"""
    __tablename__ = "rater_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    total_comparisons = Column(Integer, default=0)
    gold_task_accuracy = Column(Float, default=0.0)
    agreement_rate = Column(Float, default=0.0)
    weight = Column(Float, default=1.0)
    tier = Column(String(10), default="bronze")  # gold, silver, bronze
    last_activity = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ServiceHeartbeat(Base):
    """Service heartbeat for monitoring"""
    __tablename__ = "service_heartbeats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(50), unique=True, nullable=False, index=True)
    status = Column(String(20), default="unknown")  # healthy, degraded, down, unknown
    last_heartbeat = Column(DateTime, nullable=True)
    active_jobs = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    extra_data = Column(Text, nullable=True)  # JSON string for extra info


class VideoEloRating(Base):
    """Elo rating for each video in the lameness hierarchy"""
    __tablename__ = "video_elo_ratings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(100), unique=True, nullable=False, index=True)
    elo_rating = Column(Float, default=1500.0)
    elo_uncertainty = Column(Float, default=350.0)  # Rating deviation (like Glicko)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    ties = Column(Integer, default=0)
    total_comparisons = Column(Integer, default=0)
    win_probability = Column(Float, default=0.5)  # Cumulative win probability
    normalized_score = Column(Float, nullable=True)  # David's score normalized 0-1
    rank_position = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PairwiseComparison(Base):
    """Individual pairwise comparison record"""
    __tablename__ = "pairwise_comparisons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id_1 = Column(String(100), nullable=False, index=True)
    video_id_2 = Column(String(100), nullable=False, index=True)
    winner = Column(Integer, nullable=False)  # 1, 2, or 0 (tie)
    degree = Column(Integer, default=1)  # Strength of preference (0-3)
    confidence = Column(String(20), default="confident")
    rater_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    rater_weight = Column(Float, default=1.0)  # Weight based on rater tier
    is_gold_task = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("winner IN (0, 1, 2)", name="valid_comparison_winner"),
        CheckConstraint("degree >= 0 AND degree <= 3", name="valid_degree"),
    )


class EloHistory(Base):
    """Historical Elo rating snapshots for trend analysis"""
    __tablename__ = "elo_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(100), nullable=False, index=True)
    elo_rating = Column(Float, nullable=False)
    comparison_count = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class HierarchySnapshot(Base):
    """Snapshot of full hierarchy for reproducibility"""
    __tablename__ = "hierarchy_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    total_videos = Column(Integer, nullable=False)
    total_comparisons = Column(Integer, nullable=False)
    steepness = Column(Float, nullable=True)  # Hierarchy steepness metric
    steepness_std = Column(Float, nullable=True)
    inter_rater_reliability = Column(Float, nullable=True)  # ICC value
    ranking_data = Column(Text, nullable=False)  # JSON of full ranking
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============== DATABASE FUNCTIONS ==============

async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections"""
    await engine.dispose()
