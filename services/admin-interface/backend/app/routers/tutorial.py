"""
Tutorial Management API
Manages gold tasks for tutorial mode and rater validation
"""
import random
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.database import get_db, GoldTask, User
from app.middleware.auth import get_current_user, get_optional_user

router = APIRouter()

VIDEOS_DIR = Path("/app/data/videos")


# ============== REQUEST/RESPONSE MODELS ==============

class GoldTaskCreate(BaseModel):
    """Create a new gold task / tutorial example"""
    video_id_1: str
    video_id_2: str
    correct_winner: int = Field(..., ge=0, le=2)  # 0=tie, 1=video1, 2=video2
    correct_degree: int = Field(default=2, ge=1, le=3)  # 1-3 strength
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    description: Optional[str] = None
    hint: Optional[str] = None
    is_tutorial: bool = False
    tutorial_order: Optional[int] = None


class GoldTaskUpdate(BaseModel):
    """Update a gold task"""
    correct_winner: Optional[int] = Field(None, ge=0, le=2)
    correct_degree: Optional[int] = Field(None, ge=1, le=3)
    difficulty: Optional[str] = Field(None, pattern="^(easy|medium|hard)$")
    description: Optional[str] = None
    hint: Optional[str] = None
    is_tutorial: Optional[bool] = None
    tutorial_order: Optional[int] = None
    is_active: Optional[bool] = None


class GoldTaskResponse(BaseModel):
    """Gold task response"""
    id: str
    video_id_1: str
    video_id_2: str
    correct_winner: int
    correct_degree: int
    difficulty: str
    description: Optional[str]
    hint: Optional[str]
    is_tutorial: bool
    tutorial_order: Optional[int]
    is_active: bool
    created_at: datetime


class TutorialExample(BaseModel):
    """Tutorial example for frontend"""
    id: str
    video_id_1: str
    video_id_2: str
    description: str
    hint: str
    correct_answer: int  # -3 to 3 (7-point scale)
    difficulty: str
    order: int


# ============== TUTORIAL ENDPOINTS ==============

@router.get("/examples")
async def get_tutorial_examples(
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get all active tutorial examples in order.
    Returns tutorial tasks sorted by tutorial_order.
    """
    result = await db.execute(
        select(GoldTask)
        .where(and_(
            GoldTask.is_tutorial == True,
            GoldTask.is_active == True
        ))
        .order_by(GoldTask.tutorial_order.asc().nullslast(), GoldTask.created_at.asc())
    )
    tasks = result.scalars().all()

    examples = []
    for i, task in enumerate(tasks):
        # Convert winner/degree to 7-point scale (-3 to 3)
        if task.correct_winner == 0:
            correct_answer = 0  # Tie
        elif task.correct_winner == 1:
            correct_answer = -task.correct_degree  # Video A more lame
        else:
            correct_answer = task.correct_degree  # Video B more lame

        examples.append({
            "id": str(task.id),
            "video_id_1": task.video_id_1,
            "video_id_2": task.video_id_2,
            "description": task.description or "Compare these two videos",
            "hint": task.hint or "Look for signs of lameness",
            "correct_answer": correct_answer,
            "difficulty": task.difficulty,
            "order": task.tutorial_order or i + 1
        })

    return {
        "examples": examples,
        "total": len(examples)
    }


@router.post("/examples/auto-generate")
async def auto_generate_tutorial(
    count: int = 3,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Auto-generate tutorial examples from random video pairs.
    Admin only. Creates placeholder tutorials that need expert review.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    # Get available video IDs
    video_ids = []
    for video_file in VIDEOS_DIR.glob("*.*"):
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = video_file.stem.split("_")[0]
            video_ids.append(video_id)

    if len(video_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 videos to create tutorials"
        )

    # Get existing tutorial count for ordering
    existing_count = await db.execute(
        select(func.count(GoldTask.id)).where(GoldTask.is_tutorial == True)
    )
    start_order = (existing_count.scalar() or 0) + 1

    # Create tutorial examples
    created = []
    difficulties = ["easy", "medium", "hard"]
    descriptions = [
        "Watch for arched back - a clear sign of lameness",
        "Observe head bobbing patterns during walking",
        "Look for uneven stride length between legs",
        "Notice if the cow favors one side while walking",
        "Check for hesitation or reluctance to move",
    ]

    used_pairs = set()
    for i in range(min(count, len(video_ids) // 2)):
        # Pick random pair
        attempts = 0
        while attempts < 50:
            v1, v2 = random.sample(video_ids, 2)
            pair_key = tuple(sorted([v1, v2]))
            if pair_key not in used_pairs:
                used_pairs.add(pair_key)
                break
            attempts += 1
        else:
            continue

        task = GoldTask(
            video_id_1=v1,
            video_id_2=v2,
            correct_winner=random.choice([1, 2]),  # Placeholder - admin should review
            correct_degree=2,
            difficulty=difficulties[i % len(difficulties)],
            description=descriptions[i % len(descriptions)],
            hint="Review this tutorial and set the correct answer.",
            is_tutorial=True,
            tutorial_order=start_order + i,
            created_by=current_user.id,
            is_active=False  # Inactive until reviewed
        )
        db.add(task)
        created.append(task)

    await db.commit()

    return {
        "message": f"Created {len(created)} tutorial examples",
        "note": "Tutorial examples are inactive until reviewed. Set correct answers and activate them.",
        "created_ids": [str(t.id) for t in created]
    }


# ============== GOLD TASK CRUD ENDPOINTS ==============

@router.get("/tasks")
async def list_gold_tasks(
    is_tutorial: Optional[bool] = None,
    is_active: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    List all gold tasks / tutorial examples.
    Admin and researcher only.
    """
    if current_user.role not in ["admin", "researcher"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or researcher access required"
        )

    query = select(GoldTask)

    if is_tutorial is not None:
        query = query.where(GoldTask.is_tutorial == is_tutorial)
    if is_active is not None:
        query = query.where(GoldTask.is_active == is_active)

    query = query.order_by(GoldTask.tutorial_order.asc().nullslast(), GoldTask.created_at.desc())
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    tasks = result.scalars().all()

    # Count totals
    count_query = select(func.count(GoldTask.id))
    if is_tutorial is not None:
        count_query = count_query.where(GoldTask.is_tutorial == is_tutorial)
    if is_active is not None:
        count_query = count_query.where(GoldTask.is_active == is_active)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    return {
        "tasks": [
            {
                "id": str(t.id),
                "video_id_1": t.video_id_1,
                "video_id_2": t.video_id_2,
                "correct_winner": t.correct_winner,
                "correct_degree": t.correct_degree,
                "difficulty": t.difficulty,
                "description": t.description,
                "hint": t.hint,
                "is_tutorial": t.is_tutorial,
                "tutorial_order": t.tutorial_order,
                "is_active": t.is_active,
                "created_at": t.created_at.isoformat()
            }
            for t in tasks
        ],
        "total": total
    }


@router.post("/tasks", status_code=status.HTTP_201_CREATED)
async def create_gold_task(
    task_data: GoldTaskCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Create a new gold task / tutorial example.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    # Verify videos exist
    video1_path = VIDEOS_DIR / f"{task_data.video_id_1}.mp4"
    video2_path = VIDEOS_DIR / f"{task_data.video_id_2}.mp4"

    # Check various extensions
    video1_exists = any(
        (VIDEOS_DIR / f"{task_data.video_id_1}{ext}").exists()
        for ext in [".mp4", ".avi", ".mov", ".mkv", ""]
    )
    video2_exists = any(
        (VIDEOS_DIR / f"{task_data.video_id_2}{ext}").exists()
        for ext in [".mp4", ".avi", ".mov", ".mkv", ""]
    )

    if not video1_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video {task_data.video_id_1} not found"
        )
    if not video2_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video {task_data.video_id_2} not found"
        )

    task = GoldTask(
        video_id_1=task_data.video_id_1,
        video_id_2=task_data.video_id_2,
        correct_winner=task_data.correct_winner,
        correct_degree=task_data.correct_degree,
        difficulty=task_data.difficulty,
        description=task_data.description,
        hint=task_data.hint,
        is_tutorial=task_data.is_tutorial,
        tutorial_order=task_data.tutorial_order,
        created_by=current_user.id,
        is_active=True
    )

    db.add(task)
    await db.commit()
    await db.refresh(task)

    return {
        "id": str(task.id),
        "video_id_1": task.video_id_1,
        "video_id_2": task.video_id_2,
        "correct_winner": task.correct_winner,
        "correct_degree": task.correct_degree,
        "difficulty": task.difficulty,
        "description": task.description,
        "hint": task.hint,
        "is_tutorial": task.is_tutorial,
        "tutorial_order": task.tutorial_order,
        "is_active": task.is_active,
        "created_at": task.created_at.isoformat()
    }


@router.put("/tasks/{task_id}")
async def update_gold_task(
    task_id: str,
    task_data: GoldTaskUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Update a gold task / tutorial example.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    result = await db.execute(
        select(GoldTask).where(GoldTask.id == uuid.UUID(task_id))
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Update fields
    if task_data.correct_winner is not None:
        task.correct_winner = task_data.correct_winner
    if task_data.correct_degree is not None:
        task.correct_degree = task_data.correct_degree
    if task_data.difficulty is not None:
        task.difficulty = task_data.difficulty
    if task_data.description is not None:
        task.description = task_data.description
    if task_data.hint is not None:
        task.hint = task_data.hint
    if task_data.is_tutorial is not None:
        task.is_tutorial = task_data.is_tutorial
    if task_data.tutorial_order is not None:
        task.tutorial_order = task_data.tutorial_order
    if task_data.is_active is not None:
        task.is_active = task_data.is_active

    await db.commit()
    await db.refresh(task)

    return {
        "id": str(task.id),
        "video_id_1": task.video_id_1,
        "video_id_2": task.video_id_2,
        "correct_winner": task.correct_winner,
        "correct_degree": task.correct_degree,
        "difficulty": task.difficulty,
        "description": task.description,
        "hint": task.hint,
        "is_tutorial": task.is_tutorial,
        "tutorial_order": task.tutorial_order,
        "is_active": task.is_active,
        "created_at": task.created_at.isoformat()
    }


@router.delete("/tasks/{task_id}")
async def delete_gold_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Delete a gold task / tutorial example.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    result = await db.execute(
        select(GoldTask).where(GoldTask.id == uuid.UUID(task_id))
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    await db.delete(task)
    await db.commit()

    return {"message": "Task deleted successfully"}


@router.get("/stats")
async def get_tutorial_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Get tutorial and gold task statistics.
    """
    # Count tutorials
    tutorial_count = await db.execute(
        select(func.count(GoldTask.id)).where(GoldTask.is_tutorial == True)
    )
    total_tutorials = tutorial_count.scalar() or 0

    active_tutorial_count = await db.execute(
        select(func.count(GoldTask.id)).where(
            and_(GoldTask.is_tutorial == True, GoldTask.is_active == True)
        )
    )
    active_tutorials = active_tutorial_count.scalar() or 0

    # Count gold tasks (validation tasks)
    gold_count = await db.execute(
        select(func.count(GoldTask.id)).where(GoldTask.is_tutorial == False)
    )
    total_gold = gold_count.scalar() or 0

    active_gold_count = await db.execute(
        select(func.count(GoldTask.id)).where(
            and_(GoldTask.is_tutorial == False, GoldTask.is_active == True)
        )
    )
    active_gold = active_gold_count.scalar() or 0

    # Difficulty distribution
    difficulty_result = await db.execute(
        select(GoldTask.difficulty, func.count(GoldTask.id))
        .where(GoldTask.is_active == True)
        .group_by(GoldTask.difficulty)
    )
    difficulty_dist = dict(difficulty_result.fetchall())

    return {
        "tutorials": {
            "total": total_tutorials,
            "active": active_tutorials
        },
        "gold_tasks": {
            "total": total_gold,
            "active": active_gold
        },
        "difficulty_distribution": difficulty_dist
    }
