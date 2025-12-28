"""
Rater Reliability Service
Implements Dawid-Skene and GLAD models for computing rater ability and weighted consensus.

Key Features:
- Gold task injection and validation
- Dawid-Skene algorithm for latent truth inference
- GLAD (Generative model of Labels, Abilities, and Difficulties)
- Rater tier system (Bronze/Silver/Gold)
- Weighted consensus calculation
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict
import yaml
from shared.utils.nats_client import NATSClient


@dataclass
class RaterStats:
    """Statistics for a single rater"""
    rater_id: str
    total_comparisons: int
    gold_task_accuracy: float
    estimated_ability: float
    tier: str  # bronze, silver, gold
    weight: float
    confusion_matrix: List[List[float]]  # For Dawid-Skene
    agreement_rate: float


@dataclass
class ConsensusResult:
    """Result of weighted consensus calculation"""
    video_id: str
    estimated_label: int  # 0 = sound, 1 = lame
    probability: float
    confidence: float
    num_raters: int
    weighted_votes: Dict[str, float]
    rater_contributions: List[Dict[str, Any]]


class DawidSkene:
    """
    Dawid-Skene Algorithm for latent truth inference.
    
    Simultaneously infers:
    - True labels for items
    - Error rates (confusion matrices) for each rater
    
    Reference: Dawid & Skene (1979) "Maximum Likelihood Estimation of Observer 
               Error-Rates Using the EM Algorithm"
    """
    
    def __init__(self, num_classes: int = 2, max_iter: int = 100, tol: float = 1e-4):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.tol = tol
        
        # Model parameters
        self.pi = None  # Class priors
        self.theta = None  # Rater confusion matrices
        self.labels = None  # Estimated true labels
    
    def fit(self, annotations: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
        """
        Fit the Dawid-Skene model using EM algorithm.
        
        Args:
            annotations: Dict[item_id -> Dict[rater_id -> label]]
        
        Returns:
            estimated_labels: Dict[item_id -> label]
            rater_confusion: Dict[rater_id -> confusion_matrix]
        """
        if not annotations:
            return {}, {}
        
        items = list(annotations.keys())
        raters = set()
        for item_annotations in annotations.values():
            raters.update(item_annotations.keys())
        raters = list(raters)
        
        n_items = len(items)
        n_raters = len(raters)
        
        # Initialize with majority vote
        initial_labels = {}
        for item in items:
            votes = list(annotations[item].values())
            if votes:
                initial_labels[item] = max(set(votes), key=votes.count)
            else:
                initial_labels[item] = 0
        
        # Initialize parameters
        self.pi = np.ones(self.num_classes) / self.num_classes
        self.theta = {}
        for rater in raters:
            # Initialize with slightly biased diagonal (80% accuracy)
            cm = np.eye(self.num_classes) * 0.8 + np.ones((self.num_classes, self.num_classes)) * 0.1
            cm = cm / cm.sum(axis=1, keepdims=True)
            self.theta[rater] = cm
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step: Estimate posterior over true labels
            q = np.zeros((n_items, self.num_classes))
            
            for i, item in enumerate(items):
                for c in range(self.num_classes):
                    log_prob = np.log(self.pi[c] + 1e-10)
                    for rater, label in annotations[item].items():
                        if rater in self.theta:
                            log_prob += np.log(self.theta[rater][c, label] + 1e-10)
                    q[i, c] = log_prob
                
                # Normalize
                q[i] = np.exp(q[i] - q[i].max())
                q[i] /= q[i].sum()
            
            # M-step: Update parameters
            # Update class priors
            new_pi = q.sum(axis=0) / n_items
            
            # Update confusion matrices
            new_theta = {}
            for rater in raters:
                cm = np.zeros((self.num_classes, self.num_classes))
                for i, item in enumerate(items):
                    if rater in annotations[item]:
                        label = annotations[item][rater]
                        cm[:, label] += q[i]
                
                # Normalize rows
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm = cm / row_sums
                new_theta[rater] = cm
            
            # Check convergence
            pi_change = np.abs(new_pi - self.pi).max()
            theta_change = max(
                np.abs(new_theta[r] - self.theta[r]).max()
                for r in raters
            )
            
            self.pi = new_pi
            self.theta = new_theta
            
            if pi_change < self.tol and theta_change < self.tol:
                break
        
        # Get estimated labels
        estimated_labels = {}
        label_probs = {}
        for i, item in enumerate(items):
            estimated_labels[item] = int(q[i].argmax())
            label_probs[item] = float(q[i].max())
        
        self.labels = estimated_labels
        self.label_probs = label_probs
        
        return estimated_labels, self.theta
    
    def get_rater_ability(self, rater_id: str) -> float:
        """Get estimated ability score for a rater (diagonal average of confusion matrix)"""
        if rater_id not in self.theta:
            return 0.5
        return float(np.diag(self.theta[rater_id]).mean())


class GLAD:
    """
    GLAD: Generative model of Labels, Abilities, and Difficulties.
    
    Extends Dawid-Skene by also modeling item difficulty.
    
    Reference: Whitehill et al. (2009) "Whose Vote Should Count More: 
               Optimal Integration of Labels from Labelers of Unknown Expertise"
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol
        
        self.alpha = None  # Rater abilities
        self.beta = None   # Item difficulties (inverse)
        self.labels = None
    
    def fit(self, annotations: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """
        Fit GLAD model.
        
        Returns:
            estimated_labels: Dict[item_id -> label]
            rater_abilities: Dict[rater_id -> ability]
            item_difficulties: Dict[item_id -> difficulty]
        """
        if not annotations:
            return {}, {}, {}
        
        items = list(annotations.keys())
        raters = set()
        for item_annotations in annotations.values():
            raters.update(item_annotations.keys())
        raters = list(raters)
        
        n_items = len(items)
        n_raters = len(raters)
        
        item_to_idx = {item: i for i, item in enumerate(items)}
        rater_to_idx = {rater: i for i, rater in enumerate(raters)}
        
        # Initialize
        self.alpha = np.ones(n_raters)  # All raters start with ability 1
        self.beta = np.ones(n_items)    # All items start with difficulty 1
        
        # Initialize labels with majority vote
        z = np.zeros(n_items)
        for i, item in enumerate(items):
            votes = list(annotations[item].values())
            if votes:
                z[i] = np.mean(votes) > 0.5
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step: Update label posteriors
            z_new = np.zeros(n_items)
            
            for i, item in enumerate(items):
                log_odds = 0
                for rater, label in annotations[item].items():
                    j = rater_to_idx[rater]
                    # Sigmoid model: P(l=1|z=1) = sigmoid(alpha * beta)
                    prob_correct = 1 / (1 + np.exp(-self.alpha[j] * self.beta[i]))
                    if label == 1:
                        log_odds += np.log(prob_correct + 1e-10) - np.log(1 - prob_correct + 1e-10)
                    else:
                        log_odds += np.log(1 - prob_correct + 1e-10) - np.log(prob_correct + 1e-10)
                
                z_new[i] = 1 / (1 + np.exp(-log_odds))
            
            # M-step: Update abilities and difficulties
            # Simple gradient-based update
            for j, rater in enumerate(raters):
                correct = 0
                total = 0
                for item, labels in annotations.items():
                    if rater in labels:
                        i = item_to_idx[item]
                        expected_label = z_new[i] > 0.5
                        if labels[rater] == expected_label:
                            correct += 1
                        total += 1
                
                if total > 0:
                    accuracy = correct / total
                    # Convert to log-odds scale
                    self.alpha[j] = np.log(accuracy + 0.01) - np.log(1 - accuracy + 0.01)
            
            # Update difficulties (inverse of consensus strength)
            for i, item in enumerate(items):
                votes = list(annotations[item].values())
                if votes:
                    agreement = abs(np.mean(votes) - 0.5) * 2  # 0 to 1
                    self.beta[i] = agreement + 0.5  # Higher for easier items
            
            # Check convergence
            if np.abs(z_new - z).max() < self.tol:
                break
            z = z_new
        
        # Return results
        estimated_labels = {item: int(z[item_to_idx[item]] > 0.5) for item in items}
        rater_abilities = {rater: float(1 / (1 + np.exp(-self.alpha[rater_to_idx[rater]]))) 
                          for rater in raters}
        item_difficulties = {item: float(1 / self.beta[item_to_idx[item]]) for item in items}
        
        self.labels = estimated_labels
        
        return estimated_labels, rater_abilities, item_difficulties


class RaterReliabilityService:
    """
    Service for computing rater reliability and weighted consensus.
    """
    
    # Tier thresholds based on gold task accuracy
    TIER_THRESHOLDS = {
        'gold': 0.85,
        'silver': 0.70,
        'bronze': 0.0
    }
    
    # Weight multipliers for tiers
    TIER_WEIGHTS = {
        'gold': 1.5,
        'silver': 1.0,
        'bronze': 0.5
    }
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Data storage
        self.data_dir = Path("/app/data/rater_reliability")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.dawid_skene = DawidSkene(num_classes=2)
        self.glad = GLAD()
        
        # Rater data
        self.rater_stats: Dict[str, RaterStats] = {}
        self.gold_tasks: Dict[str, int] = {}  # video_id -> true_label
        self.annotations: Dict[str, Dict[str, int]] = {}  # video_id -> {rater_id -> label}
        
        self._load_data()
    
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_data(self):
        """Load existing data from disk"""
        gold_path = self.data_dir / "gold_tasks.json"
        if gold_path.exists():
            with open(gold_path) as f:
                self.gold_tasks = json.load(f)
        
        annotations_path = self.data_dir / "annotations.json"
        if annotations_path.exists():
            with open(annotations_path) as f:
                self.annotations = json.load(f)
        
        rater_stats_path = self.data_dir / "rater_stats.json"
        if rater_stats_path.exists():
            with open(rater_stats_path) as f:
                data = json.load(f)
                self.rater_stats = {
                    k: RaterStats(**v) for k, v in data.items()
                }
    
    def _save_data(self):
        """Save data to disk"""
        with open(self.data_dir / "gold_tasks.json", "w") as f:
            json.dump(self.gold_tasks, f)
        
        with open(self.data_dir / "annotations.json", "w") as f:
            json.dump(self.annotations, f)
        
        with open(self.data_dir / "rater_stats.json", "w") as f:
            json.dump({k: asdict(v) for k, v in self.rater_stats.items()}, f)
    
    def add_gold_task(self, video_id: str, true_label: int):
        """Add a gold task (expert-verified label)"""
        self.gold_tasks[video_id] = true_label
        self._save_data()
    
    def record_annotation(self, video_id: str, rater_id: str, label: int):
        """Record a new annotation from a rater"""
        if video_id not in self.annotations:
            self.annotations[video_id] = {}
        
        self.annotations[video_id][rater_id] = label
        self._save_data()
    
    def compute_gold_task_accuracy(self, rater_id: str) -> float:
        """Compute accuracy on gold tasks for a rater"""
        correct = 0
        total = 0
        
        for video_id, true_label in self.gold_tasks.items():
            if video_id in self.annotations:
                if rater_id in self.annotations[video_id]:
                    if self.annotations[video_id][rater_id] == true_label:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.5
    
    def determine_tier(self, gold_accuracy: float) -> str:
        """Determine rater tier based on gold task accuracy"""
        if gold_accuracy >= self.TIER_THRESHOLDS['gold']:
            return 'gold'
        elif gold_accuracy >= self.TIER_THRESHOLDS['silver']:
            return 'silver'
        else:
            return 'bronze'
    
    def update_rater_stats(self):
        """Update statistics for all raters"""
        # Get all raters
        all_raters = set()
        for video_annotations in self.annotations.values():
            all_raters.update(video_annotations.keys())
        
        # Run Dawid-Skene to get confusion matrices
        if self.annotations:
            _, confusion_matrices = self.dawid_skene.fit(self.annotations)
        else:
            confusion_matrices = {}
        
        # Compute stats for each rater
        for rater_id in all_raters:
            total_comparisons = sum(
                1 for anns in self.annotations.values() if rater_id in anns
            )
            
            gold_accuracy = self.compute_gold_task_accuracy(rater_id)
            tier = self.determine_tier(gold_accuracy)
            
            # Get ability from Dawid-Skene or GLAD
            if rater_id in confusion_matrices:
                estimated_ability = self.dawid_skene.get_rater_ability(rater_id)
                cm = confusion_matrices[rater_id].tolist()
            else:
                estimated_ability = 0.5 + (gold_accuracy - 0.5) * 0.5  # Simple estimate
                cm = [[0.5, 0.5], [0.5, 0.5]]
            
            # Compute agreement rate with other raters
            agreements = 0
            opportunities = 0
            for video_id, anns in self.annotations.items():
                if rater_id in anns:
                    other_votes = [l for r, l in anns.items() if r != rater_id]
                    if other_votes:
                        majority = max(set(other_votes), key=other_votes.count)
                        if anns[rater_id] == majority:
                            agreements += 1
                        opportunities += 1
            
            agreement_rate = agreements / opportunities if opportunities > 0 else 0.5
            
            # Compute weight
            weight = self.TIER_WEIGHTS[tier] * estimated_ability
            
            self.rater_stats[rater_id] = RaterStats(
                rater_id=rater_id,
                total_comparisons=total_comparisons,
                gold_task_accuracy=gold_accuracy,
                estimated_ability=estimated_ability,
                tier=tier,
                weight=weight,
                confusion_matrix=cm,
                agreement_rate=agreement_rate
            )
        
        self._save_data()
    
    def compute_weighted_consensus(self, video_id: str) -> Optional[ConsensusResult]:
        """Compute weighted consensus for a video"""
        if video_id not in self.annotations:
            return None
        
        annotations = self.annotations[video_id]
        if not annotations:
            return None
        
        # Update rater stats first
        self.update_rater_stats()
        
        # Compute weighted votes
        weighted_votes = {'lame': 0.0, 'sound': 0.0}
        rater_contributions = []
        
        for rater_id, label in annotations.items():
            weight = self.rater_stats.get(rater_id, RaterStats(
                rater_id=rater_id, total_comparisons=1, gold_task_accuracy=0.5,
                estimated_ability=0.5, tier='bronze', weight=0.5,
                confusion_matrix=[[0.5, 0.5], [0.5, 0.5]], agreement_rate=0.5
            )).weight
            
            if label == 1:
                weighted_votes['lame'] += weight
            else:
                weighted_votes['sound'] += weight
            
            rater_contributions.append({
                'rater_id': rater_id,
                'label': label,
                'weight': weight,
                'tier': self.rater_stats.get(rater_id, RaterStats(
                    rater_id=rater_id, total_comparisons=1, gold_task_accuracy=0.5,
                    estimated_ability=0.5, tier='bronze', weight=0.5,
                    confusion_matrix=[[0.5, 0.5], [0.5, 0.5]], agreement_rate=0.5
                )).tier
            })
        
        total_weight = weighted_votes['lame'] + weighted_votes['sound']
        if total_weight == 0:
            probability = 0.5
        else:
            probability = weighted_votes['lame'] / total_weight
        
        estimated_label = 1 if probability > 0.5 else 0
        confidence = abs(probability - 0.5) * 2  # 0 to 1
        
        return ConsensusResult(
            video_id=video_id,
            estimated_label=estimated_label,
            probability=probability,
            confidence=confidence,
            num_raters=len(annotations),
            weighted_votes=weighted_votes,
            rater_contributions=rater_contributions
        )
    
    async def handle_comparison_submitted(self, data: dict):
        """Handle a new comparison submission"""
        video_id_1 = data.get("video_id_1")
        video_id_2 = data.get("video_id_2")
        rater_id = data.get("rater_id", "anonymous")
        winner = data.get("winner")  # 1 = video_1, 2 = video_2, 0 = tie
        
        # For pairwise, we record relative labels
        # Video that "won" (is more lame) gets label 1
        if winner == 1:
            self.record_annotation(video_id_1, rater_id, 1)
            self.record_annotation(video_id_2, rater_id, 0)
        elif winner == 2:
            self.record_annotation(video_id_1, rater_id, 0)
            self.record_annotation(video_id_2, rater_id, 1)
        # Ties don't contribute to binary labels
        
        # Update and publish
        self.update_rater_stats()
        
        # Publish rater reliability update
        if rater_id in self.rater_stats:
            stats = self.rater_stats[rater_id]
            await self.nats_client.publish(
                self.config.get("nats", {}).get("subjects", {}).get(
                    "rater_reliability_updated", "rater.reliability.updated"
                ),
                {
                    "rater_id": rater_id,
                    "tier": stats.tier,
                    "weight": stats.weight,
                    "gold_accuracy": stats.gold_task_accuracy,
                    "total_comparisons": stats.total_comparisons
                }
            )
    
    def get_all_rater_stats(self) -> List[Dict]:
        """Get stats for all raters"""
        return [asdict(stats) for stats in self.rater_stats.values()]
    
    def get_consensus_for_all_videos(self) -> List[Dict]:
        """Get consensus results for all videos"""
        results = []
        for video_id in self.annotations.keys():
            consensus = self.compute_weighted_consensus(video_id)
            if consensus:
                results.append(asdict(consensus))
        return results
    
    async def start(self):
        """Start the rater reliability service"""
        await self.nats_client.connect()
        
        # Subscribe to comparison submissions
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "hitl_comparison_submitted", "hitl.comparison.submitted"
        )
        print(f"Rater Reliability Service subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.handle_comparison_submitted)
        
        print("=" * 60)
        print("Rater Reliability Service Started")
        print("=" * 60)
        print(f"Gold tasks: {len(self.gold_tasks)}")
        print(f"Tracked raters: {len(self.rater_stats)}")
        print(f"Tier thresholds: {self.TIER_THRESHOLDS}")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    service = RaterReliabilityService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())

