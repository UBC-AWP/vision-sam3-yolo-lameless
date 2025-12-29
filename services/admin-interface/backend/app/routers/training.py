"""
Training endpoints
"""
import os
import json
import nats
import random
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from itertools import combinations

router = APIRouter()

TRAINING_DIR = Path("/app/data/training")
RESULTS_DIR = Path("/app/data/results")
VIDEOS_DIR = Path("/app/data/videos")
PAIRWISE_DIR = TRAINING_DIR / "pairwise"

# NATS connection
nats_client = None


async def get_nats():
    """Get NATS connection"""
    global nats_client
    if nats_client is None or not nats_client.is_connected:
        nats_url = os.getenv("NATS_URL", "nats://nats:4222")
        nats_client = await nats.connect(nats_url)
    return nats_client


class LabelRequest(BaseModel):
    label: int  # 0 = sound, 1 = lame
    confidence: Optional[str] = "certain"  # certain, uncertain


@router.post("/videos/{video_id}/label")
async def label_video(video_id: str, label_request: LabelRequest):
    """Submit label for a video"""
    # Store label
    labels_dir = TRAINING_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    label_file = labels_dir / f"{video_id}_label.json"
    
    label_data = {
        "video_id": video_id,
        "label": label_request.label,
        "confidence": label_request.confidence,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(label_file, "w") as f:
        json.dump(label_data, f)
    
    # Publish new label event to trigger auto-training check
    try:
        nc = await get_nats()
        await nc.publish(
            "training.data.added",
            json.dumps({
                "video_id": video_id,
                "label": label_request.label,
                "timestamp": datetime.utcnow().isoformat()
            }).encode()
        )
    except Exception as e:
        print(f"Failed to publish label event: {e}")
    
    return {
        "video_id": video_id,
        "label": label_request.label,
        "status": "saved"
    }


@router.get("/queue")
async def get_training_queue():
    """Get videos that need labeling (active learning queue)"""
    videos = []
    
    fusion_dir = RESULTS_DIR / "fusion"
    labels_dir = TRAINING_DIR / "labels"
    
    if fusion_dir.exists():
        for fusion_file in fusion_dir.glob("*_fusion.json"):
            video_id = fusion_file.stem.replace("_fusion", "")
            label_file = labels_dir / f"{video_id}_label.json"
            
            if not label_file.exists():
                with open(fusion_file) as f:
                    fusion_data = json.load(f)
                    fusion_result = fusion_data.get("fusion_result", {})
                    
                    # Prioritize uncertain predictions
                    prob = fusion_result.get("final_probability", 0.5)
                    uncertainty = abs(0.5 - prob)  # Lower uncertainty = more uncertain
                    
                    videos.append({
                        "video_id": video_id,
                        "predicted_probability": prob,
                        "uncertainty": uncertainty
                    })
    
    # Sort by uncertainty (most uncertain first)
    videos.sort(key=lambda x: x["uncertainty"])
    
    return {
        "videos": videos[:50],  # Top 50 most uncertain
        "total": len(videos)
    }


@router.get("/stats")
async def get_training_stats():
    """Get training dataset statistics"""
    labels_dir = TRAINING_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    total_labels = 0
    sound_count = 0
    lame_count = 0
    
    for label_file in labels_dir.glob("*_label.json"):
        try:
            with open(label_file) as f:
                label_data = json.load(f)
                total_labels += 1
                if label_data.get("label") == 0:
                    sound_count += 1
                elif label_data.get("label") == 1:
                    lame_count += 1
        except:
            pass
    
    return {
        "total_labels": total_labels,
        "sound_count": sound_count,
        "lame_count": lame_count,
        "balance_ratio": sound_count / lame_count if lame_count > 0 else None,
        "ready_for_training": total_labels >= 10 and sound_count > 0 and lame_count > 0
    }


@router.get("/status")
async def get_training_status():
    """Get training job status"""
    status_file = TRAINING_DIR / "training_status.json"
    
    if status_file.exists():
        try:
            with open(status_file) as f:
                return json.load(f)
        except:
            pass
    
    return {
        "status": "idle",
        "last_trained": None,
        "samples_used": 0,
        "metrics": {},
        "models": []
    }


@router.post("/ml/start")
async def start_ml_training():
    """Trigger ML training manually"""
    try:
        nc = await get_nats()
        await nc.publish(
            "training.ml.requested",
            json.dumps({
                "requested_at": datetime.utcnow().isoformat(),
                "manual": True
            }).encode()
        )
        
        return {
            "status": "training_requested",
            "message": "ML training request sent. Training will start shortly."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send training request: {e}")


@router.post("/yolo/start")
async def start_yolo_training():
    """Trigger YOLO training"""
    try:
        nc = await get_nats()
        await nc.publish(
            "training.yolo.requested",
            json.dumps({
                "requested_at": datetime.utcnow().isoformat(),
                "manual": True
            }).encode()
        )
        
        return {
            "status": "training_requested",
            "message": "YOLO training request sent."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send training request: {e}")


@router.get("/models")
async def get_trained_models():
    """List trained models"""
    models_dir = TRAINING_DIR / "models"
    models = []
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.joblib"):
            stat = model_file.stat()
            models.append({
                "name": model_file.stem,
                "file": model_file.name,
                "size_kb": stat.st_size / 1024,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    return {
        "models": models,
        "total": len(models)
    }


# ==================== Pairwise Comparison Endpoints ====================

class PairwiseComparisonRequest(BaseModel):
    video_id_1: str
    video_id_2: str
    winner: int  # 1 = video_1 more lame, 2 = video_2 more lame, 0 = equal
    confidence: str = "confident"  # very_confident, confident, uncertain


@router.post("/pairwise")
async def submit_pairwise_comparison(comparison: PairwiseComparisonRequest):
    """Submit a pairwise comparison result"""
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create comparison ID
    pair_key = f"{min(comparison.video_id_1, comparison.video_id_2)}_{max(comparison.video_id_1, comparison.video_id_2)}"
    
    comparison_data = {
        "video_id_1": comparison.video_id_1,
        "video_id_2": comparison.video_id_2,
        "winner": comparison.winner,
        "confidence": comparison.confidence,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Load existing comparisons for this pair
    comparison_file = PAIRWISE_DIR / f"{pair_key}.json"
    comparisons = []
    if comparison_file.exists():
        with open(comparison_file) as f:
            data = json.load(f)
            comparisons = data.get("comparisons", [])
    
    comparisons.append(comparison_data)
    
    # Save updated comparisons
    with open(comparison_file, "w") as f:
        json.dump({
            "pair_key": pair_key,
            "video_id_1": comparison.video_id_1,
            "video_id_2": comparison.video_id_2,
            "comparisons": comparisons
        }, f, indent=2)
    
    return {
        "status": "saved",
        "pair_key": pair_key,
        "total_comparisons": len(comparisons)
    }


@router.get("/pairwise/next")
async def get_next_pairwise(exclude_completed: bool = True):
    """Get the next video pair to compare"""
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all video IDs
    video_ids = []
    for video_file in VIDEOS_DIR.glob("*.*"):
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = video_file.stem.split("_")[0]
            video_ids.append(video_id)
    
    if len(video_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 videos for pairwise comparison")
    
    # Generate all possible pairs
    all_pairs = list(combinations(sorted(video_ids), 2))
    
    # Get completed pairs
    completed_pairs = set()
    if exclude_completed:
        for pair_file in PAIRWISE_DIR.glob("*.json"):
            pair_key = pair_file.stem
            completed_pairs.add(pair_key)
    
    # Find pairs that haven't been compared yet
    pending_pairs = []
    for v1, v2 in all_pairs:
        pair_key = f"{v1}_{v2}"
        if pair_key not in completed_pairs:
            pending_pairs.append((v1, v2))
    
    if not pending_pairs:
        return {
            "status": "all_completed",
            "total_pairs": len(all_pairs),
            "completed_pairs": len(completed_pairs)
        }
    
    # Select a random pending pair
    video_id_1, video_id_2 = random.choice(pending_pairs)
    
    # Randomly swap order to avoid bias
    if random.random() > 0.5:
        video_id_1, video_id_2 = video_id_2, video_id_1
    
    return {
        "video_id_1": video_id_1,
        "video_id_2": video_id_2,
        "pending_pairs": len(pending_pairs),
        "total_pairs": len(all_pairs),
        "completed_pairs": len(completed_pairs)
    }


@router.get("/pairwise/stats")
async def get_pairwise_stats():
    """Get pairwise comparison statistics"""
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
    
    total_comparisons = 0
    pairs_compared = 0
    
    for pair_file in PAIRWISE_DIR.glob("*.json"):
        with open(pair_file) as f:
            data = json.load(f)
            comparisons = data.get("comparisons", [])
            total_comparisons += len(comparisons)
            pairs_compared += 1
    
    # Count total possible pairs
    video_ids = []
    for video_file in VIDEOS_DIR.glob("*.*"):
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = video_file.stem.split("_")[0]
            video_ids.append(video_id)
    
    total_possible_pairs = len(list(combinations(video_ids, 2))) if len(video_ids) >= 2 else 0
    
    return {
        "total_comparisons": total_comparisons,
        "pairs_compared": pairs_compared,
        "total_possible_pairs": total_possible_pairs,
        "completion_rate": pairs_compared / total_possible_pairs if total_possible_pairs > 0 else 0
    }


@router.get("/pairwise/ranking")
async def get_elo_ranking():
    """Calculate and return Elo-based lameness ranking"""
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Elo ratings
    elo_ratings = {}
    K = 32  # Elo K-factor
    
    # Process all comparisons
    all_comparisons = []
    for pair_file in PAIRWISE_DIR.glob("*.json"):
        with open(pair_file) as f:
            data = json.load(f)
            for comp in data.get("comparisons", []):
                all_comparisons.append(comp)
    
    # Initialize ratings for all videos
    video_ids = set()
    for comp in all_comparisons:
        video_ids.add(comp["video_id_1"])
        video_ids.add(comp["video_id_2"])
    
    for vid in video_ids:
        elo_ratings[vid] = 1500  # Starting Elo
    
    # Process comparisons chronologically
    all_comparisons.sort(key=lambda x: x.get("timestamp", ""))
    
    for comp in all_comparisons:
        v1, v2 = comp["video_id_1"], comp["video_id_2"]
        winner = comp["winner"]
        
        if v1 not in elo_ratings:
            elo_ratings[v1] = 1500
        if v2 not in elo_ratings:
            elo_ratings[v2] = 1500
        
        r1, r2 = elo_ratings[v1], elo_ratings[v2]
        
        # Expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores (1 = v1 more lame, 2 = v2 more lame, 0 = tie)
        if winner == 1:
            s1, s2 = 1, 0
        elif winner == 2:
            s1, s2 = 0, 1
        else:
            s1, s2 = 0.5, 0.5
        
        # Update ratings
        elo_ratings[v1] = r1 + K * (s1 - e1)
        elo_ratings[v2] = r2 + K * (s2 - e2)
    
    # Create ranking (higher Elo = more lame)
    ranking = [
        {"video_id": vid, "elo_rating": round(rating, 1), "rank": 0}
        for vid, rating in elo_ratings.items()
    ]
    ranking.sort(key=lambda x: x["elo_rating"], reverse=True)
    
    # Assign ranks
    for i, item in enumerate(ranking):
        item["rank"] = i + 1
    
    return {
        "ranking": ranking,
        "total_videos": len(ranking),
        "total_comparisons": len(all_comparisons)
    }


# ==================== Triplet Comparison Endpoints ====================

TRIPLET_DIR = TRAINING_DIR / "triplet"


class TripletComparisonRequest(BaseModel):
    reference_id: str
    comparison_a_id: str
    comparison_b_id: str
    selected_answer: str  # 'A' or 'B'
    confidence: str = "medium"  # high, medium, low
    task_type: str = "similarity"  # similarity or dissimilarity


@router.get("/triplet/next")
async def get_next_triplet():
    """Get the next triplet task to compare"""
    TRIPLET_DIR.mkdir(parents=True, exist_ok=True)

    # Get all video IDs
    video_ids = []
    for video_file in VIDEOS_DIR.glob("*.*"):
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = video_file.stem.split("_")[0]
            if video_id not in video_ids:
                video_ids.append(video_id)

    if len(video_ids) < 3:
        return {
            "status": "insufficient_videos",
            "message": "Need at least 3 videos for triplet comparison",
            "pending_tasks": 0,
            "total_tasks": 0
        }

    # Generate all possible triplets (reference, A, B)
    # We use combinations to avoid duplicates
    from itertools import permutations
    all_triplets = []
    for ref in video_ids:
        others = [v for v in video_ids if v != ref]
        for i, a in enumerate(others):
            for b in others[i+1:]:
                # Create canonical triplet key (sorted A,B to avoid duplicates)
                triplet_key = f"{ref}_{min(a,b)}_{max(a,b)}"
                all_triplets.append({
                    "key": triplet_key,
                    "reference": ref,
                    "a": min(a, b),
                    "b": max(a, b)
                })

    # Remove duplicate triplet keys
    seen_keys = set()
    unique_triplets = []
    for t in all_triplets:
        if t["key"] not in seen_keys:
            seen_keys.add(t["key"])
            unique_triplets.append(t)

    # Get completed triplets
    completed_triplets = set()
    for triplet_file in TRIPLET_DIR.glob("*.json"):
        triplet_key = triplet_file.stem
        completed_triplets.add(triplet_key)

    # Find pending triplets
    pending_triplets = [t for t in unique_triplets if t["key"] not in completed_triplets]

    if not pending_triplets:
        return {
            "status": "all_completed",
            "pending_tasks": 0,
            "total_tasks": len(unique_triplets)
        }

    # Select a random pending triplet
    selected = random.choice(pending_triplets)

    # Randomly decide task type (similarity or dissimilarity)
    task_type = random.choice(["similarity", "dissimilarity"])

    # Randomly swap A and B to avoid position bias
    a, b = selected["a"], selected["b"]
    if random.random() > 0.5:
        a, b = b, a

    return {
        "reference_id": selected["reference"],
        "comparison_a_id": a,
        "comparison_b_id": b,
        "task_type": task_type,
        "pending_tasks": len(pending_triplets),
        "total_tasks": len(unique_triplets)
    }


@router.post("/triplet")
async def submit_triplet_comparison(comparison: TripletComparisonRequest):
    """Submit a triplet comparison result"""
    TRIPLET_DIR.mkdir(parents=True, exist_ok=True)

    # Create triplet key (canonical form)
    a, b = comparison.comparison_a_id, comparison.comparison_b_id
    triplet_key = f"{comparison.reference_id}_{min(a,b)}_{max(a,b)}"

    comparison_data = {
        "reference_id": comparison.reference_id,
        "comparison_a_id": comparison.comparison_a_id,
        "comparison_b_id": comparison.comparison_b_id,
        "selected_answer": comparison.selected_answer,
        "confidence": comparison.confidence,
        "task_type": comparison.task_type,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Load existing comparisons for this triplet
    triplet_file = TRIPLET_DIR / f"{triplet_key}.json"
    comparisons = []
    if triplet_file.exists():
        with open(triplet_file) as f:
            data = json.load(f)
            comparisons = data.get("comparisons", [])

    comparisons.append(comparison_data)

    # Save updated comparisons
    with open(triplet_file, "w") as f:
        json.dump({
            "triplet_key": triplet_key,
            "reference_id": comparison.reference_id,
            "comparison_a_id": min(a, b),
            "comparison_b_id": max(a, b),
            "comparisons": comparisons
        }, f, indent=2)

    return {
        "status": "saved",
        "triplet_key": triplet_key,
        "total_comparisons": len(comparisons)
    }


@router.get("/triplet/stats")
async def get_triplet_stats():
    """Get triplet comparison statistics"""
    TRIPLET_DIR.mkdir(parents=True, exist_ok=True)

    total_comparisons = 0
    triplets_completed = 0
    similarity_tasks = 0
    dissimilarity_tasks = 0

    for triplet_file in TRIPLET_DIR.glob("*.json"):
        with open(triplet_file) as f:
            data = json.load(f)
            comparisons = data.get("comparisons", [])
            total_comparisons += len(comparisons)
            triplets_completed += 1

            for comp in comparisons:
                if comp.get("task_type") == "similarity":
                    similarity_tasks += 1
                else:
                    dissimilarity_tasks += 1

    # Count total possible triplets
    video_ids = []
    for video_file in VIDEOS_DIR.glob("*.*"):
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = video_file.stem.split("_")[0]
            if video_id not in video_ids:
                video_ids.append(video_id)

    # Calculate total possible triplets
    n = len(video_ids)
    if n >= 3:
        # For each reference, we can have C(n-1, 2) pairs
        from math import comb
        total_possible = n * comb(n - 1, 2)
    else:
        total_possible = 0

    return {
        "total_comparisons": total_comparisons,
        "completed_tasks": triplets_completed,
        "total_tasks": total_possible,
        "similarity_tasks": similarity_tasks,
        "dissimilarity_tasks": dissimilarity_tasks,
        "completion_rate": triplets_completed / total_possible if total_possible > 0 else 0
    }


# ==================== Training Progress & Leaderboard Endpoints ====================

PROGRESS_DIR = TRAINING_DIR / "progress"


class TrainingProgressRequest(BaseModel):
    total_score: int
    total_attempts: int
    correct_count: int
    current_level: int
    streak: int
    rater_tier: Optional[str] = None


@router.get("/learn/progress")
async def get_training_progress(user_id: Optional[str] = None):
    """Get user's training progress"""
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    # Use a default user_id if not provided (for anonymous users)
    if not user_id:
        user_id = "anonymous"

    progress_file = PROGRESS_DIR / f"{user_id}.json"

    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)

    # Return default progress
    return {
        "user_id": user_id,
        "total_score": 0,
        "total_attempts": 0,
        "correct_count": 0,
        "current_level": 1,
        "streak": 0,
        "rater_tier": "bronze",
        "updated_at": None
    }


@router.put("/learn/progress")
async def save_training_progress(progress: TrainingProgressRequest, user_id: Optional[str] = None):
    """Save user's training progress"""
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    if not user_id:
        user_id = "anonymous"

    progress_data = {
        "user_id": user_id,
        "total_score": progress.total_score,
        "total_attempts": progress.total_attempts,
        "correct_count": progress.correct_count,
        "current_level": progress.current_level,
        "streak": progress.streak,
        "rater_tier": progress.rater_tier,
        "updated_at": datetime.utcnow().isoformat()
    }

    progress_file = PROGRESS_DIR / f"{user_id}.json"
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

    return {"status": "saved", **progress_data}


@router.get("/learn/leaderboard")
async def get_leaderboard(limit: int = 20):
    """Get training leaderboard (top users by score)"""
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    leaderboard = []

    for progress_file in PROGRESS_DIR.glob("*.json"):
        try:
            with open(progress_file) as f:
                data = json.load(f)
                leaderboard.append({
                    "user_id": data.get("user_id", progress_file.stem),
                    "username": data.get("username", data.get("user_id", progress_file.stem)),
                    "total_score": data.get("total_score", 0),
                    "accuracy": data.get("correct_count", 0) / max(data.get("total_attempts", 1), 1),
                    "rater_tier": data.get("rater_tier", "bronze"),
                    "current_level": data.get("current_level", 1)
                })
        except Exception as e:
            print(f"Error reading progress file {progress_file}: {e}")

    # Sort by score descending
    leaderboard.sort(key=lambda x: x["total_score"], reverse=True)

    # Assign ranks
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    return {
        "leaderboard": leaderboard[:limit],
        "total_users": len(leaderboard)
    }


@router.get("/learn/examples")
async def get_training_examples(difficulty: Optional[str] = None):
    """Get training examples from tutorial/gold tasks"""
    # Try to get examples from the tutorial system (gold_tasks table via tutorial router)
    examples = {
        "easy": [],
        "medium": [],
        "hard": []
    }

    # Read from gold_tasks directory
    gold_tasks_dir = TRAINING_DIR / "gold_tasks"
    gold_tasks_dir.mkdir(parents=True, exist_ok=True)

    for task_file in gold_tasks_dir.glob("*.json"):
        try:
            with open(task_file) as f:
                task = json.load(f)
                if task.get("is_active", True):
                    diff = task.get("difficulty", "medium")
                    if diff not in examples:
                        diff = "medium"

                    examples[diff].append({
                        "id": task.get("id", task_file.stem),
                        "video_id_1": task.get("video_id_1"),
                        "video_id_2": task.get("video_id_2"),
                        "description": task.get("description", "Compare these two cows"),
                        "hint": task.get("hint", "Look at the gait patterns"),
                        "correct_winner": task.get("correct_winner", 0),
                        "correct_degree": task.get("correct_degree", 1),
                        "difficulty": diff
                    })
        except Exception as e:
            print(f"Error reading task file {task_file}: {e}")

    # If no tasks exist, create some default examples using available videos
    if all(len(v) == 0 for v in examples.values()):
        video_ids = []
        for video_file in VIDEOS_DIR.glob("*.*"):
            if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                video_id = video_file.stem.split("_")[0]
                if video_id not in video_ids:
                    video_ids.append(video_id)

        # Create synthetic examples if we have videos
        if len(video_ids) >= 2:
            for i, diff in enumerate(["easy", "medium", "hard"]):
                # Create 3 examples per difficulty
                for j in range(min(3, len(video_ids) - 1)):
                    idx1 = (i * 3 + j) % len(video_ids)
                    idx2 = (idx1 + 1) % len(video_ids)
                    examples[diff].append({
                        "id": f"auto_{diff}_{j}",
                        "video_id_1": video_ids[idx1],
                        "video_id_2": video_ids[idx2],
                        "description": f"Compare these two cows ({diff} difficulty)",
                        "hint": "Watch their walking patterns carefully",
                        "correct_winner": random.choice([0, 1, 2]),  # Random for auto-generated
                        "correct_degree": random.choice([1, 2, 3]),
                        "difficulty": diff,
                        "is_auto_generated": True
                    })

    if difficulty and difficulty in examples:
        return {
            "examples": examples[difficulty],
            "total": len(examples[difficulty])
        }

    return {
        "examples": examples,
        "total": sum(len(v) for v in examples.values())
    }
