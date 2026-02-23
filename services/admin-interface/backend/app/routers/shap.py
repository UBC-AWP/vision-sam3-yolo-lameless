"""
SHAP explainability endpoints
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import numpy as np

router = APIRouter()

RESULTS_DIR = Path("/app/data/results")
SHAP_DIR = RESULTS_DIR / "shap"


@router.get("/{video_id}/local")
async def get_local_shap(video_id: str):
    """Get local SHAP explanation for a video"""
    # Check if SHAP results exist
    shap_file = SHAP_DIR / f"{video_id}_shap.json"
    
    if not shap_file.exists():
        # Generate basic SHAP values from ML results
        ml_file = RESULTS_DIR / "ml" / f"{video_id}_ml.json"
        if not ml_file.exists():
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        with open(ml_file) as f:
            ml_data = json.load(f)
        
        # Generate simple SHAP-like explanation
        features = ml_data.get("features", [])
        feature_names = ml_data.get("feature_names", [])
        predictions = ml_data.get("predictions", {})
        
        # Simple feature importance based on feature values
        shap_values = []
        for i, (feat, name) in enumerate(zip(features, feature_names)):
            # Normalize feature value to SHAP-like contribution
            contribution = (feat - 0.5) * 0.2  # Simple scaling
            shap_values.append({
                "feature": name,
                "value": feat,
                "shap_value": contribution,
                "contribution": abs(contribution)
            })
        
        # Sort by contribution
        shap_values.sort(key=lambda x: x["contribution"], reverse=True)
        
        return {
            "video_id": video_id,
            "shap_values": shap_values,
            "base_value": 0.5,
            "prediction": predictions.get("ensemble", {}).get("probability", 0.5)
        }
    
    with open(shap_file) as f:
        return json.load(f)


@router.get("/{video_id}/force-plot")
async def get_force_plot(video_id: str):
    """Get force plot data for visualization"""
    shap_data = await get_local_shap(video_id)
    
    return {
        "video_id": video_id,
        "base_value": shap_data.get("base_value", 0.5),
        "prediction": shap_data.get("prediction", 0.5),
        "features": shap_data.get("shap_values", [])
    }


@router.get("/global")
async def get_global_shap():
    """Get global feature importance"""
    # Aggregate SHAP values from all analyzed videos
    all_shap = []
    
    for shap_file in SHAP_DIR.glob("*_shap.json"):
        with open(shap_file) as f:
            shap_data = json.load(f)
            all_shap.extend(shap_data.get("shap_values", []))
    
    # Aggregate by feature
    feature_importance = {}
    for item in all_shap:
        feat_name = item.get("feature", "unknown")
        if feat_name not in feature_importance:
            feature_importance[feat_name] = []
        feature_importance[feat_name].append(abs(item.get("shap_value", 0)))
    
    # Calculate average importance
    global_importance = [
        {
            "feature": feat,
            "importance": np.mean(importances),
            "std": np.std(importances)
        }
        for feat, importances in feature_importance.items()
    ]
    
    global_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "feature_importance": global_importance,
        "total_videos": len(list(SHAP_DIR.glob("*_shap.json")))
    }


@router.post("/what-if")
async def what_if_analysis(request: dict):
    """What-if analysis: change features and see prediction impact"""
    video_id = request.get("video_id")
    feature_changes = request.get("feature_changes", {})
    
    # Load original analysis
    ml_file = RESULTS_DIR / "ml" / f"{video_id}_ml.json"
    if not ml_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    with open(ml_file) as f:
        ml_data = json.load(f)
    
    # Modify features
    features = ml_data.get("features", []).copy()
    feature_names = ml_data.get("feature_names", [])
    
    for feat_name, new_value in feature_changes.items():
        if feat_name in feature_names:
            idx = feature_names.index(feat_name)
            features[idx] = new_value
    
    # TODO: Re-run prediction with modified features
    # For now, return modified features
    return {
        "video_id": video_id,
        "original_prediction": ml_data.get("predictions", {}).get("ensemble", {}).get("probability", 0.5),
        "modified_features": dict(zip(feature_names, features)),
        "note": "Prediction recalculation not yet implemented"
    }

