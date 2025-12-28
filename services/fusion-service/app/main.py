"""
Fusion Service
Combines predictions from all pipelines to generate final lameness score.

Enhanced with:
- TCN, Transformer, and Graph Transformer predictions
- Human consensus integration
- Rule-based gating and stacking meta-model
- Confidence calibration
- Detailed pipeline comparison report
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import yaml
from shared.utils.nats_client import NATSClient


class FusionService:
    """
    Enhanced Fusion service combining all pipeline predictions.
    
    Pipelines integrated:
    - ML (XGBoost, CatBoost, LightGBM ensemble)
    - TCN (Temporal Convolutional Network)
    - Transformer (Gait Transformer)
    - GNN (Graph Transformer / GraphGPS)
    - Human consensus (weighted by rater reliability)
    """
    
    # Pipeline weights for weighted average fusion
    PIPELINE_WEIGHTS = {
        "ml": 0.25,
        "tcn": 0.15,
        "transformer": 0.15,
        "gnn": 0.15,
        "human": 0.30  # High weight for human consensus
    }
    
    # Confidence thresholds for gating
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    LOW_CONFIDENCE_THRESHOLD = 0.55
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Model storage
        self.models_dir = Path("/app/shared/models/fusion")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load stacking meta-model if available
        self.stacking_model = None
        self._load_stacking_model()
        
        # Directories
        self.results_dir = Path("/app/data/results/fusion")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for pipeline results
        self.pipeline_results = {}
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_stacking_model(self):
        """Load stacking meta-model if available"""
        stacking_file = self.models_dir / "stacking_model.pkl"
        if stacking_file.exists():
            try:
                import pickle
                with open(stacking_file, "rb") as f:
                    self.stacking_model = pickle.load(f)
                print(f"✅ Loaded stacking model: {stacking_file}")
            except Exception as e:
                print(f"⚠️ Failed to load stacking model: {e}")
    
    def collect_pipeline_predictions(self, video_id: str) -> Dict[str, Any]:
        """Collect predictions from all pipelines including new DL models"""
        predictions = {}
        
        # ML pipeline predictions (XGBoost, CatBoost, LightGBM)
        ml_file = Path(f"/app/data/results/ml/{video_id}_ml.json")
        if ml_file.exists():
            with open(ml_file) as f:
                ml_data = json.load(f)
                if "predictions" in ml_data:
                    predictions["ml"] = {
                        "probability": ml_data["predictions"].get("ensemble", {}).get("probability", 0.5),
                        "uncertainty": 0.1,  # Default uncertainty
                        "model_predictions": ml_data["predictions"]
                    }
        
        # TCN pipeline predictions
        tcn_file = Path(f"/app/data/results/tcn/{video_id}_tcn.json")
        if tcn_file.exists():
            with open(tcn_file) as f:
                tcn_data = json.load(f)
                predictions["tcn"] = {
                    "probability": tcn_data.get("severity_score", 0.5),
                    "uncertainty": tcn_data.get("uncertainty", 0.1)
                }
        
        # Transformer pipeline predictions
        transformer_file = Path(f"/app/data/results/transformer/{video_id}_transformer.json")
        if transformer_file.exists():
            with open(transformer_file) as f:
                transformer_data = json.load(f)
                predictions["transformer"] = {
                    "probability": transformer_data.get("severity_score", 0.5),
                    "uncertainty": transformer_data.get("uncertainty", 0.1),
                    "temporal_saliency": transformer_data.get("temporal_saliency", [])
                }
        
        # GNN (GraphGPS) pipeline predictions
        gnn_file = Path(f"/app/data/results/gnn/{video_id}_gnn.json")
        if gnn_file.exists():
            with open(gnn_file) as f:
                gnn_data = json.load(f)
                predictions["gnn"] = {
                    "probability": gnn_data.get("severity_score", 0.5),
                    "uncertainty": gnn_data.get("uncertainty", 0.1),
                    "neighbor_influence": gnn_data.get("neighbor_influence", [])
                }
        
        # Human consensus (from rater reliability service)
        human_file = Path(f"/app/data/rater_reliability/consensus/{video_id}.json")
        if human_file.exists():
            with open(human_file) as f:
                human_data = json.load(f)
                predictions["human"] = {
                    "probability": human_data.get("probability", 0.5),
                    "confidence": human_data.get("confidence", 0.5),
                    "num_raters": human_data.get("num_raters", 0)
                }
        
        # Also load feature-level data for SHAP
        # YOLO features
        yolo_file = Path(f"/app/data/results/yolo/{video_id}_yolo.json")
        if yolo_file.exists():
            with open(yolo_file) as f:
                yolo_data = json.load(f)
                if "features" in yolo_data:
                    predictions["yolo"] = yolo_data["features"]
        
        # T-LEAP features
        tleap_file = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
        if tleap_file.exists():
            with open(tleap_file) as f:
                tleap_data = json.load(f)
                predictions["tleap"] = tleap_data.get("locomotion_features", {})
        
        return predictions
    
    def apply_gating_rules(self, predictions: Dict[str, Any]) -> Tuple[str, str]:
        """
        Apply rule-based gating to determine fusion strategy.
        
        Returns:
            decision_mode: 'human', 'automated', 'hybrid', 'uncertain'
            explanation: Reason for the decision mode
        """
        human_pred = predictions.get("human", {})
        human_conf = human_pred.get("confidence", 0)
        human_num_raters = human_pred.get("num_raters", 0)
        
        # Collect automated predictions
        auto_preds = []
        for key in ["ml", "tcn", "transformer", "gnn"]:
            if key in predictions:
                auto_preds.append(predictions[key].get("probability", 0.5))
        
        if not auto_preds:
            if human_num_raters > 0:
                return "human", "No automated predictions available; using human consensus"
            return "uncertain", "Insufficient data from all sources"
        
        auto_mean = np.mean(auto_preds)
        auto_std = np.std(auto_preds)
        auto_agreement = 1.0 - auto_std  # Higher when models agree
        
        # Rule 1: High human confidence with sufficient raters
        if human_conf >= self.HIGH_CONFIDENCE_THRESHOLD and human_num_raters >= 3:
            return "human", f"High human consensus confidence ({human_conf:.2f}) with {human_num_raters} raters"
        
        # Rule 2: High model agreement with high confidence
        if auto_agreement >= 0.9 and all(
            abs(p - 0.5) > 0.3 for p in auto_preds
        ):
            return "automated", f"Strong model agreement ({auto_agreement:.2f}) with high confidence"
        
        # Rule 3: Model disagreement - request more human labels
        if auto_std > 0.25:
            return "uncertain", f"Model disagreement (std={auto_std:.2f}); more human labels recommended"
        
        # Rule 4: Hybrid approach for moderate cases
        return "hybrid", "Moderate confidence; combining human and automated predictions"
    
    def fuse_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced fusion combining all pipelines with gating rules.
        """
        # Apply gating rules
        decision_mode, gate_explanation = self.apply_gating_rules(predictions)
        
        # Collect pipeline probabilities and uncertainties
        pipeline_probs = {}
        pipeline_uncertainties = {}
        
        for key in ["ml", "tcn", "transformer", "gnn", "human"]:
            if key in predictions:
                pipeline_probs[key] = predictions[key].get("probability", 0.5)
                pipeline_uncertainties[key] = predictions[key].get("uncertainty", 
                    1.0 - predictions[key].get("confidence", 0.5))
        
        # Determine fusion probability based on decision mode
        if decision_mode == "human" and "human" in pipeline_probs:
            fusion_prob = pipeline_probs["human"]
            confidence = predictions["human"].get("confidence", 0.5)
        
        elif decision_mode == "automated":
            # Use stacking model if available
            if self.stacking_model:
                features = [pipeline_probs.get(k, 0.5) for k in ["ml", "tcn", "transformer", "gnn"]]
                try:
                    fusion_prob = float(self.stacking_model.predict_proba([features])[0, 1])
                except:
                    fusion_prob = np.mean(list(pipeline_probs.values()))
            else:
                # Weighted average of automated pipelines
                weighted_sum = 0.0
                total_weight = 0.0
                for key in ["ml", "tcn", "transformer", "gnn"]:
                    if key in pipeline_probs:
                        weight = self.PIPELINE_WEIGHTS.get(key, 0.1)
                        # Reduce weight for high uncertainty
                        uncertainty = pipeline_uncertainties.get(key, 0.5)
                        adjusted_weight = weight * (1.0 - uncertainty * 0.5)
                        weighted_sum += pipeline_probs[key] * adjusted_weight
                        total_weight += adjusted_weight
                
                fusion_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            # Compute confidence from agreement
            auto_probs = [v for k, v in pipeline_probs.items() if k != "human"]
            confidence = 1.0 - np.std(auto_probs) if auto_probs else 0.5
        
        elif decision_mode == "hybrid":
            # Combine human and automated with configured weights
            weighted_sum = 0.0
            total_weight = 0.0
            
            for key, prob in pipeline_probs.items():
                weight = self.PIPELINE_WEIGHTS.get(key, 0.1)
                uncertainty = pipeline_uncertainties.get(key, 0.5)
                adjusted_weight = weight * (1.0 - uncertainty * 0.5)
                weighted_sum += prob * adjusted_weight
                total_weight += adjusted_weight
            
            fusion_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
            confidence = 1.0 - np.std(list(pipeline_probs.values()))
        
        else:  # uncertain
            fusion_prob = 0.5
            confidence = 0.0
        
        # Compute agreement metrics
        all_probs = list(pipeline_probs.values())
        model_agreement = 1.0 - np.std(all_probs) if all_probs else 0.0
        all_predictions = [int(p > 0.5) for p in all_probs]
        unanimous = len(set(all_predictions)) == 1 if all_predictions else False
        
        # Determine recommendation
        if confidence < 0.3 or decision_mode == "uncertain":
            recommendation = "Request more human labels for this video"
        elif fusion_prob > 0.7:
            recommendation = "High lameness probability - consider veterinary examination"
        elif fusion_prob < 0.3:
            recommendation = "Low lameness probability - monitor routine"
        else:
            recommendation = "Moderate lameness indication - continue observation"
        
        return {
            "final_probability": float(fusion_prob),
            "final_prediction": int(fusion_prob > 0.5),
            "confidence": float(confidence),
            "decision_mode": decision_mode,
            "gate_explanation": gate_explanation,
            "model_agreement": float(model_agreement),
            "unanimous": unanimous,
            "recommendation": recommendation,
            "pipeline_contributions": {
                key: {
                    "probability": float(pipeline_probs.get(key, 0.5)),
                    "uncertainty": float(pipeline_uncertainties.get(key, 0.5)),
                    "prediction": int(pipeline_probs.get(key, 0.5) > 0.5),
                    "weight": self.PIPELINE_WEIGHTS.get(key, 0.1)
                }
                for key in ["ml", "tcn", "transformer", "gnn", "human"]
                if key in pipeline_probs
            },
            "pipelines_used": list(pipeline_probs.keys()),
            "tleap_features": predictions.get("tleap", {}),
            "yolo_features": predictions.get("yolo", {})
        }
    
    async def process_video(self, video_data: dict):
        """Process video through fusion service"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"Fusion service processing video {video_id}")
        
        try:
            # Collect predictions from all pipelines
            predictions = self.collect_pipeline_predictions(video_id)
            
            if not predictions:
                print(f"No pipeline predictions found for {video_id}")
                return
            
            # Fuse predictions
            fusion_result = self.fuse_predictions(predictions)
            
            # Save results
            results = {
                "video_id": video_id,
                "fusion_result": fusion_result,
                "pipeline_predictions": predictions,
                "timestamp": video_data.get("timestamp", "")
            }
            
            results_file = self.results_dir / f"{video_id}_fusion.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish analysis complete event
            analysis_result = {
                "video_id": video_id,
                "final_probability": fusion_result["final_probability"],
                "final_prediction": fusion_result["final_prediction"],
                "results_path": str(results_file),
                "pipeline_contributions": fusion_result["pipeline_contributions"]
            }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["analysis_complete"],
                analysis_result
            )
            
            print(f"Fusion service completed for {video_id}")
            
        except Exception as e:
            print(f"Error in fusion service for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the fusion service"""
        await self.nats_client.connect()
        
        # Subscribe to ML pipeline results (last in sequence)
        subject = self.config["nats"]["subjects"]["pipeline_ml"]
        print(f"Fusion service subscribed to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("Fusion service started. Waiting for pipeline results...")
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    service = FusionService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())

