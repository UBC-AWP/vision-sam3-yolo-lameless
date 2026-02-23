"""
EloSteepness Ranking System
Based on research paper methodology for lameness hierarchy construction.

Implements:
- Bayesian Elo rating with uncertainty
- David's Score calculation
- Steepness metric for hierarchy linearity
- Inter-rater reliability (ICC approximation)
- Comparison weighting by rater tier and degree
"""
import json
import math
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from itertools import combinations
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.dialects.postgresql import insert

from app.database import (
    get_db, VideoEloRating, PairwiseComparison,
    EloHistory, HierarchySnapshot, User, RaterStats
)
from app.middleware.auth import get_current_user, get_optional_user

router = APIRouter()

# Constants
INITIAL_ELO = 1500.0
INITIAL_UNCERTAINTY = 350.0
K_FACTOR_BASE = 32
MIN_K_FACTOR = 10
VIDEOS_DIR = Path("/app/data/videos")
PAIRWISE_DIR = Path("/app/data/training/pairwise")


class ComparisonSubmission(BaseModel):
    video_id_1: str
    video_id_2: str
    winner: int  # 0=tie, 1=video_1, 2=video_2
    degree: int = 1  # 0-3 strength of preference
    confidence: str = "confident"
    raw_score: Optional[int] = None  # -3 to 3 from UI


class EloCalculator:
    """
    Elo rating calculator with Bayesian updates.
    Based on EloSteepness package methodology.
    """

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    @staticmethod
    def dynamic_k_factor(
        uncertainty: float,
        games_played: int,
        degree: int = 1,
        rater_weight: float = 1.0
    ) -> float:
        """
        Calculate dynamic K-factor based on:
        - Rating uncertainty (higher uncertainty = larger updates)
        - Number of games (fewer games = larger updates)
        - Comparison degree (stronger preference = larger update)
        - Rater weight (gold raters have more influence)
        """
        # Base K adjusted by uncertainty
        k = K_FACTOR_BASE * (uncertainty / INITIAL_UNCERTAINTY)

        # Reduce K as more games are played
        games_factor = max(0.5, 1.0 - (games_played / 100))
        k *= games_factor

        # Degree multiplier (degree 0=tie gets 0.5, degree 3 gets 1.5)
        degree_factor = 0.5 + (degree * 0.33)
        k *= degree_factor

        # Rater weight (gold=1.5, silver=1.0, bronze=0.75)
        k *= rater_weight

        return max(MIN_K_FACTOR, k)

    @staticmethod
    def update_ratings(
        rating_a: float,
        rating_b: float,
        uncertainty_a: float,
        uncertainty_b: float,
        games_a: int,
        games_b: int,
        winner: int,  # 0=tie, 1=A wins, 2=B wins
        degree: int = 1,
        rater_weight: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """
        Update ratings for both players after a comparison.
        Returns: (new_rating_a, new_rating_b, new_uncertainty_a, new_uncertainty_b)
        """
        # Expected scores
        expected_a = EloCalculator.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        # Actual scores
        if winner == 1:  # A wins (more lame)
            actual_a, actual_b = 1.0, 0.0
        elif winner == 2:  # B wins (more lame)
            actual_a, actual_b = 0.0, 1.0
        else:  # Tie
            actual_a, actual_b = 0.5, 0.5

        # Dynamic K-factors
        k_a = EloCalculator.dynamic_k_factor(uncertainty_a, games_a, degree, rater_weight)
        k_b = EloCalculator.dynamic_k_factor(uncertainty_b, games_b, degree, rater_weight)

        # Update ratings
        new_rating_a = rating_a + k_a * (actual_a - expected_a)
        new_rating_b = rating_b + k_b * (actual_b - expected_b)

        # Update uncertainties (decrease with more games)
        decay = 0.98
        new_uncertainty_a = max(50, uncertainty_a * decay)
        new_uncertainty_b = max(50, uncertainty_b * decay)

        return new_rating_a, new_rating_b, new_uncertainty_a, new_uncertainty_b


class DavidsScoreCalculator:
    """
    Calculate David's Score for dominance hierarchy.
    Based on methodology from EloSteepness package.
    """

    @staticmethod
    def calculate_scores(comparisons: List[Dict]) -> Dict[str, float]:
        """
        Calculate David's Scores from pairwise comparisons.

        DS_i = w_i + w2_i - l_i - l2_i
        where:
        - w_i = proportion of wins for individual i
        - w2_i = weighted wins (wins over individuals that also win a lot)
        - l_i = proportion of losses
        - l2_i = weighted losses
        """
        # Build win matrix
        video_ids = set()
        for comp in comparisons:
            video_ids.add(comp['video_id_1'])
            video_ids.add(comp['video_id_2'])

        video_list = sorted(list(video_ids))
        n = len(video_list)
        if n == 0:
            return {}

        idx = {v: i for i, v in enumerate(video_list)}

        # Win counts matrix (W_ij = wins of i over j)
        wins = [[0.0 for _ in range(n)] for _ in range(n)]
        total = [[0.0 for _ in range(n)] for _ in range(n)]

        for comp in comparisons:
            i = idx[comp['video_id_1']]
            j = idx[comp['video_id_2']]
            weight = comp.get('rater_weight', 1.0)
            degree = comp.get('degree', 1)

            # Weight by degree (stronger preference = more weight)
            w = weight * (1 + degree * 0.5)

            if comp['winner'] == 1:
                wins[i][j] += w
            elif comp['winner'] == 2:
                wins[j][i] += w
            else:  # Tie
                wins[i][j] += w * 0.5
                wins[j][i] += w * 0.5

            total[i][j] += w
            total[j][i] += w

        # Calculate proportions
        P = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if total[i][j] > 0:
                    P[i][j] = wins[i][j] / total[i][j]

        # Calculate w and l (sum of proportions)
        w = [sum(P[i][j] for j in range(n) if j != i) for i in range(n)]
        l = [sum(P[j][i] for j in range(n) if j != i) for i in range(n)]

        # Calculate w2 and l2 (weighted sums)
        w2 = [sum(P[i][j] * w[j] for j in range(n) if j != i) for i in range(n)]
        l2 = [sum(P[j][i] * l[j] for j in range(n) if j != i) for i in range(n)]

        # David's Scores
        ds = {video_list[i]: w[i] + w2[i] - l[i] - l2[i] for i in range(n)}

        # Normalize to 0-1 range
        if ds:
            min_ds = min(ds.values())
            max_ds = max(ds.values())
            if max_ds > min_ds:
                ds = {k: (v - min_ds) / (max_ds - min_ds) for k, v in ds.items()}

        return ds


class SteepnessCalculator:
    """
    Calculate hierarchy steepness from David's Scores.
    Steepness measures how linear/transitive the hierarchy is.
    """

    @staticmethod
    def calculate_steepness(normalized_scores: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate steepness using linear regression.
        Steepness = slope of regression line of DS vs rank.

        Returns: (steepness, standard_error)
        """
        if len(normalized_scores) < 3:
            return 0.0, 0.0

        # Sort by score
        sorted_items = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_items)

        # Ranks (1 to n)
        ranks = list(range(1, n + 1))
        scores = [item[1] for item in sorted_items]

        # Linear regression
        mean_rank = sum(ranks) / n
        mean_score = sum(scores) / n

        numerator = sum((r - mean_rank) * (s - mean_score) for r, s in zip(ranks, scores))
        denominator = sum((r - mean_rank) ** 2 for r in ranks)

        if denominator == 0:
            return 0.0, 0.0

        slope = numerator / denominator

        # Standard error
        y_pred = [mean_score + slope * (r - mean_rank) for r in ranks]
        ss_res = sum((s - p) ** 2 for s, p in zip(scores, y_pred))
        se = math.sqrt(ss_res / (n - 2)) / math.sqrt(denominator) if n > 2 else 0.0

        # Normalize steepness to 0-1 range
        # Maximum possible slope is -1/(n-1) for perfect linearity
        max_slope = -1.0 / (n - 1)
        steepness = abs(slope / max_slope) if max_slope != 0 else 0.0
        steepness = min(1.0, steepness)

        return steepness, se


class InterRaterReliability:
    """
    Calculate inter-rater reliability approximation.
    Uses agreement rate as a proxy for ICC.
    """

    @staticmethod
    def calculate_agreement(comparisons: List[Dict]) -> float:
        """
        Calculate agreement rate between raters on the same pairs.
        Returns a value between 0 and 1.
        """
        # Group comparisons by pair
        pair_ratings = defaultdict(list)
        for comp in comparisons:
            pair_key = tuple(sorted([comp['video_id_1'], comp['video_id_2']]))
            pair_ratings[pair_key].append(comp['winner'])

        if not pair_ratings:
            return 0.0

        # Calculate agreement for pairs with multiple ratings
        agreements = []
        for pair_key, ratings in pair_ratings.items():
            if len(ratings) > 1:
                # Calculate pairwise agreement
                n = len(ratings)
                agree_count = 0
                total_pairs = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        total_pairs += 1
                        if ratings[i] == ratings[j]:
                            agree_count += 1
                        # Partial agreement for ties vs wins
                        elif ratings[i] == 0 or ratings[j] == 0:
                            agree_count += 0.5

                if total_pairs > 0:
                    agreements.append(agree_count / total_pairs)

        return sum(agreements) / len(agreements) if agreements else 0.0


# ==================== API ENDPOINTS ====================

@router.post("/comparison")
async def submit_comparison(
    submission: ComparisonSubmission,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Submit a pairwise comparison and update Elo ratings.
    Tracks which user submitted the comparison for user-specific filtering.
    """
    # Convert raw_score (-3 to 3) to winner and degree
    if submission.raw_score is not None:
        if submission.raw_score < 0:
            submission.winner = 1
            submission.degree = abs(submission.raw_score)
        elif submission.raw_score > 0:
            submission.winner = 2
            submission.degree = submission.raw_score
        else:
            submission.winner = 0
            submission.degree = 0

    # Get or create video ratings
    async def get_or_create_rating(video_id: str) -> VideoEloRating:
        result = await db.execute(
            select(VideoEloRating).where(VideoEloRating.video_id == video_id)
        )
        rating = result.scalar_one_or_none()
        if not rating:
            rating = VideoEloRating(
                video_id=video_id,
                elo_rating=INITIAL_ELO,
                elo_uncertainty=INITIAL_UNCERTAINTY
            )
            db.add(rating)
            await db.flush()
        return rating

    rating_1 = await get_or_create_rating(submission.video_id_1)
    rating_2 = await get_or_create_rating(submission.video_id_2)

    # Get rater weight based on user tier
    rater_weight = 1.0
    rater_id = None
    if current_user:
        rater_id = current_user.id
        # Weight by rater tier
        tier_weights = {"gold": 1.5, "silver": 1.0, "bronze": 0.75}
        rater_weight = tier_weights.get(current_user.rater_tier, 1.0)

    # Update Elo ratings
    new_r1, new_r2, new_u1, new_u2 = EloCalculator.update_ratings(
        rating_1.elo_rating,
        rating_2.elo_rating,
        rating_1.elo_uncertainty,
        rating_2.elo_uncertainty,
        rating_1.total_comparisons,
        rating_2.total_comparisons,
        submission.winner,
        submission.degree,
        rater_weight
    )

    # Update rating records
    rating_1.elo_rating = new_r1
    rating_1.elo_uncertainty = new_u1
    rating_1.total_comparisons += 1
    if submission.winner == 1:
        rating_1.wins += 1
        rating_2.losses += 1
    elif submission.winner == 2:
        rating_2.wins += 1
        rating_1.losses += 1
    else:
        rating_1.ties += 1
        rating_2.ties += 1

    rating_2.elo_rating = new_r2
    rating_2.elo_uncertainty = new_u2
    rating_2.total_comparisons += 1

    # Save comparison record with rater_id
    comparison = PairwiseComparison(
        video_id_1=submission.video_id_1,
        video_id_2=submission.video_id_2,
        winner=submission.winner,
        degree=submission.degree,
        confidence=submission.confidence,
        rater_id=rater_id,
        rater_weight=rater_weight
    )
    db.add(comparison)

    # Save Elo history
    for rating in [rating_1, rating_2]:
        history = EloHistory(
            video_id=rating.video_id,
            elo_rating=rating.elo_rating,
            comparison_count=rating.total_comparisons
        )
        db.add(history)

    await db.commit()

    return {
        "status": "saved",
        "video_1": {
            "video_id": submission.video_id_1,
            "new_elo": round(new_r1, 1),
            "change": round(new_r1 - (rating_1.elo_rating - (new_r1 - rating_1.elo_rating)), 1)
        },
        "video_2": {
            "video_id": submission.video_id_2,
            "new_elo": round(new_r2, 1),
            "change": round(new_r2 - (rating_2.elo_rating - (new_r2 - rating_2.elo_rating)), 1)
        }
    }


@router.get("/hierarchy")
async def get_hierarchy(db: AsyncSession = Depends(get_db)):
    """
    Get complete lameness hierarchy with Elo ratings and David's Scores.
    """
    # Get all ratings
    result = await db.execute(
        select(VideoEloRating).order_by(VideoEloRating.elo_rating.desc())
    )
    ratings = result.scalars().all()

    # Get all comparisons for David's Score calculation
    comp_result = await db.execute(select(PairwiseComparison))
    comparisons = comp_result.scalars().all()

    # Calculate David's Scores
    comp_dicts = [
        {
            'video_id_1': c.video_id_1,
            'video_id_2': c.video_id_2,
            'winner': c.winner,
            'degree': c.degree,
            'rater_weight': c.rater_weight
        }
        for c in comparisons
    ]

    davids_scores = DavidsScoreCalculator.calculate_scores(comp_dicts)

    # Calculate steepness
    steepness, steepness_se = SteepnessCalculator.calculate_steepness(davids_scores)

    # Calculate inter-rater agreement
    agreement = InterRaterReliability.calculate_agreement(comp_dicts)

    # Build ranking
    ranking = []
    for i, rating in enumerate(ratings):
        ranking.append({
            "rank": i + 1,
            "video_id": rating.video_id,
            "elo_rating": round(rating.elo_rating, 1),
            "elo_uncertainty": round(rating.elo_uncertainty, 1),
            "davids_score": round(davids_scores.get(rating.video_id, 0.5), 4),
            "wins": rating.wins,
            "losses": rating.losses,
            "ties": rating.ties,
            "total_comparisons": rating.total_comparisons,
            "win_rate": round(rating.wins / rating.total_comparisons, 3) if rating.total_comparisons > 0 else 0
        })

    return {
        "ranking": ranking,
        "total_videos": len(ranking),
        "total_comparisons": len(comparisons),
        "metrics": {
            "steepness": round(steepness, 4),
            "steepness_se": round(steepness_se, 4),
            "inter_rater_agreement": round(agreement, 4),
            "hierarchy_linearity": "Strong" if steepness > 0.7 else "Moderate" if steepness > 0.4 else "Weak"
        }
    }


@router.get("/next-pair")
async def get_next_pair(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Get the next video pair to compare using intelligent selection.
    If user is logged in, only returns pairs they haven't rated yet.

    Prioritizes:
    1. Pairs that the current user hasn't compared yet
    2. Pairs with high uncertainty
    3. Pairs with similar ratings (harder to distinguish)
    """
    # Get all video IDs
    video_ids = []
    for video_file in VIDEOS_DIR.glob("*.*"):
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = video_file.stem.split("_")[0]
            video_ids.append(video_id)

    if len(video_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 videos")

    # Get comparisons - filter by user if logged in
    if current_user:
        # Get pairs this user has already rated
        result = await db.execute(
            select(PairwiseComparison).where(PairwiseComparison.rater_id == current_user.id)
        )
        user_comparisons = result.scalars().all()

        user_compared_pairs = set()
        for comp in user_comparisons:
            pair_key = tuple(sorted([comp.video_id_1, comp.video_id_2]))
            user_compared_pairs.add(pair_key)
    else:
        user_compared_pairs = set()

    # Get all comparisons for global stats
    all_result = await db.execute(select(PairwiseComparison))
    all_comparisons = all_result.scalars().all()

    global_compared_pairs = set()
    for comp in all_comparisons:
        pair_key = tuple(sorted([comp.video_id_1, comp.video_id_2]))
        global_compared_pairs.add(pair_key)

    # Generate all possible pairs
    all_pairs = list(combinations(sorted(video_ids), 2))

    # Find pairs this user hasn't rated (or all uncompared if not logged in)
    if current_user:
        uncompared = [p for p in all_pairs if p not in user_compared_pairs]
    else:
        uncompared = [p for p in all_pairs if p not in global_compared_pairs]

    if not uncompared:
        # User has rated all pairs
        return {
            "status": "all_completed",
            "message": "You have rated all available video pairs!" if current_user else "All pairs have been compared",
            "total_pairs": len(all_pairs),
            "completed_pairs": len(user_compared_pairs) if current_user else len(global_compared_pairs),
            "user_completed": len(user_compared_pairs) if current_user else None
        }

    # Get ratings for intelligent selection
    rating_result = await db.execute(select(VideoEloRating))
    ratings = {r.video_id: r for r in rating_result.scalars().all()}

    # Score pairs by uncertainty and rating closeness
    def pair_score(pair):
        v1, v2 = pair
        r1 = ratings.get(v1)
        r2 = ratings.get(v2)

        if not r1 or not r2:
            return float('inf')  # Prioritize unrated videos

        # Prefer pairs with similar ratings (harder comparisons)
        rating_diff = abs(r1.elo_rating - r2.elo_rating)

        # Prefer pairs with high uncertainty
        uncertainty = r1.elo_uncertainty + r2.elo_uncertainty

        return rating_diff - uncertainty * 0.5

    # Select pair (mix of random and intelligent)
    if random.random() < 0.3:
        # 30% random selection
        selected = random.choice(uncompared)
    else:
        # 70% intelligent selection
        uncompared.sort(key=pair_score)
        selected = uncompared[0]

    v1, v2 = selected
    # Randomize order
    if random.random() > 0.5:
        v1, v2 = v2, v1

    return {
        "video_id_1": v1,
        "video_id_2": v2,
        "pending_pairs": len(uncompared),
        "total_pairs": len(all_pairs),
        "completed_pairs": len(user_compared_pairs) if current_user else len(global_compared_pairs),
        "global_completed": len(global_compared_pairs),
        "user_id": str(current_user.id) if current_user else None
    }


@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Get comprehensive pairwise comparison statistics, including user-specific stats if logged in."""
    # Count comparisons
    comp_count = await db.execute(select(func.count(PairwiseComparison.id)))
    total_comparisons = comp_count.scalar() or 0

    # Count unique pairs
    unique_pairs_result = await db.execute(
        select(
            func.count(func.distinct(
                func.concat(
                    func.least(PairwiseComparison.video_id_1, PairwiseComparison.video_id_2),
                    '_',
                    func.greatest(PairwiseComparison.video_id_1, PairwiseComparison.video_id_2)
                )
            ))
        )
    )
    unique_pairs = unique_pairs_result.scalar() or 0

    # Count videos
    video_count = await db.execute(select(func.count(VideoEloRating.id)))
    total_videos = video_count.scalar() or 0

    # Get video count from directory
    video_ids = set()
    for video_file in VIDEOS_DIR.glob("*.*"):
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = video_file.stem.split("_")[0]
            video_ids.add(video_id)

    total_possible_pairs = len(list(combinations(video_ids, 2))) if len(video_ids) >= 2 else 0

    # Winner distribution
    winner_dist = await db.execute(
        select(
            PairwiseComparison.winner,
            func.count(PairwiseComparison.id)
        ).group_by(PairwiseComparison.winner)
    )
    winner_counts = dict(winner_dist.fetchall())

    # User-specific stats if logged in
    user_stats = None
    if current_user:
        # Count user's comparisons
        user_comp_count = await db.execute(
            select(func.count(PairwiseComparison.id)).where(
                PairwiseComparison.rater_id == current_user.id
            )
        )
        user_total = user_comp_count.scalar() or 0

        # Count user's unique pairs
        user_comparisons_result = await db.execute(
            select(PairwiseComparison).where(
                PairwiseComparison.rater_id == current_user.id
            )
        )
        user_comparisons = user_comparisons_result.scalars().all()
        user_unique_pairs = set()
        for comp in user_comparisons:
            pair_key = tuple(sorted([comp.video_id_1, comp.video_id_2]))
            user_unique_pairs.add(pair_key)

        user_stats = {
            "user_id": str(current_user.id),
            "username": current_user.username,
            "tier": current_user.rater_tier,
            "total_comparisons": user_total,
            "unique_pairs_compared": len(user_unique_pairs),
            "completion_rate": len(user_unique_pairs) / total_possible_pairs if total_possible_pairs > 0 else 0,
            "pending_pairs": total_possible_pairs - len(user_unique_pairs)
        }

    return {
        "total_comparisons": total_comparisons,
        "unique_pairs_compared": unique_pairs,
        "total_videos": max(total_videos, len(video_ids)),
        "total_possible_pairs": total_possible_pairs,
        "completion_rate": unique_pairs / total_possible_pairs if total_possible_pairs > 0 else 0,
        "winner_distribution": {
            "video_1_wins": winner_counts.get(1, 0),
            "video_2_wins": winner_counts.get(2, 0),
            "ties": winner_counts.get(0, 0)
        },
        "comparisons_per_pair": total_comparisons / unique_pairs if unique_pairs > 0 else 0,
        "user_stats": user_stats
    }


@router.post("/snapshot")
async def create_snapshot(
    name: str,
    description: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Create a snapshot of the current hierarchy for reproducibility."""
    # Get current hierarchy
    hierarchy = await get_hierarchy(db)

    snapshot = HierarchySnapshot(
        name=name,
        description=description,
        total_videos=hierarchy["total_videos"],
        total_comparisons=hierarchy["total_comparisons"],
        steepness=hierarchy["metrics"]["steepness"],
        steepness_std=hierarchy["metrics"]["steepness_se"],
        inter_rater_reliability=hierarchy["metrics"]["inter_rater_agreement"],
        ranking_data=json.dumps(hierarchy["ranking"])
    )

    db.add(snapshot)
    await db.commit()
    await db.refresh(snapshot)

    return {
        "id": str(snapshot.id),
        "name": snapshot.name,
        "created_at": snapshot.created_at.isoformat(),
        "total_videos": snapshot.total_videos
    }


@router.get("/snapshots")
async def list_snapshots(db: AsyncSession = Depends(get_db)):
    """List all hierarchy snapshots."""
    result = await db.execute(
        select(HierarchySnapshot).order_by(HierarchySnapshot.created_at.desc())
    )
    snapshots = result.scalars().all()

    return {
        "snapshots": [
            {
                "id": str(s.id),
                "name": s.name,
                "description": s.description,
                "total_videos": s.total_videos,
                "total_comparisons": s.total_comparisons,
                "steepness": s.steepness,
                "inter_rater_reliability": s.inter_rater_reliability,
                "created_at": s.created_at.isoformat()
            }
            for s in snapshots
        ]
    }


@router.get("/snapshot/{snapshot_id}")
async def get_snapshot(snapshot_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific hierarchy snapshot."""
    from uuid import UUID
    result = await db.execute(
        select(HierarchySnapshot).where(HierarchySnapshot.id == UUID(snapshot_id))
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return {
        "id": str(snapshot.id),
        "name": snapshot.name,
        "description": snapshot.description,
        "total_videos": snapshot.total_videos,
        "total_comparisons": snapshot.total_comparisons,
        "steepness": snapshot.steepness,
        "steepness_std": snapshot.steepness_std,
        "inter_rater_reliability": snapshot.inter_rater_reliability,
        "ranking": json.loads(snapshot.ranking_data),
        "created_at": snapshot.created_at.isoformat()
    }


@router.get("/video/{video_id}/history")
async def get_video_history(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get Elo rating history for a specific video."""
    result = await db.execute(
        select(EloHistory)
        .where(EloHistory.video_id == video_id)
        .order_by(EloHistory.recorded_at)
    )
    history = result.scalars().all()

    return {
        "video_id": video_id,
        "history": [
            {
                "elo_rating": h.elo_rating,
                "comparison_count": h.comparison_count,
                "recorded_at": h.recorded_at.isoformat()
            }
            for h in history
        ]
    }


@router.post("/recalculate")
async def recalculate_ratings(db: AsyncSession = Depends(get_db)):
    """
    Recalculate all Elo ratings from scratch using all comparisons.
    Useful after parameter changes or to ensure consistency.
    """
    # Reset all ratings
    await db.execute(
        VideoEloRating.__table__.update().values(
            elo_rating=INITIAL_ELO,
            elo_uncertainty=INITIAL_UNCERTAINTY,
            wins=0,
            losses=0,
            ties=0,
            total_comparisons=0
        )
    )

    # Get all comparisons in chronological order
    result = await db.execute(
        select(PairwiseComparison).order_by(PairwiseComparison.created_at)
    )
    comparisons = result.scalars().all()

    # Get all ratings
    ratings_result = await db.execute(select(VideoEloRating))
    ratings = {r.video_id: r for r in ratings_result.scalars().all()}

    # Process each comparison
    for comp in comparisons:
        if comp.video_id_1 not in ratings:
            ratings[comp.video_id_1] = VideoEloRating(
                video_id=comp.video_id_1,
                elo_rating=INITIAL_ELO,
                elo_uncertainty=INITIAL_UNCERTAINTY
            )
            db.add(ratings[comp.video_id_1])

        if comp.video_id_2 not in ratings:
            ratings[comp.video_id_2] = VideoEloRating(
                video_id=comp.video_id_2,
                elo_rating=INITIAL_ELO,
                elo_uncertainty=INITIAL_UNCERTAINTY
            )
            db.add(ratings[comp.video_id_2])

        r1 = ratings[comp.video_id_1]
        r2 = ratings[comp.video_id_2]

        new_r1, new_r2, new_u1, new_u2 = EloCalculator.update_ratings(
            r1.elo_rating, r2.elo_rating,
            r1.elo_uncertainty, r2.elo_uncertainty,
            r1.total_comparisons, r2.total_comparisons,
            comp.winner, comp.degree, comp.rater_weight
        )

        r1.elo_rating = new_r1
        r1.elo_uncertainty = new_u1
        r1.total_comparisons += 1

        r2.elo_rating = new_r2
        r2.elo_uncertainty = new_u2
        r2.total_comparisons += 1

        if comp.winner == 1:
            r1.wins += 1
            r2.losses += 1
        elif comp.winner == 2:
            r2.wins += 1
            r1.losses += 1
        else:
            r1.ties += 1
            r2.ties += 1

    await db.commit()

    return {
        "status": "recalculated",
        "total_comparisons": len(comparisons),
        "total_videos": len(ratings)
    }
