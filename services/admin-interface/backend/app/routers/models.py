"""
Model configuration endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import json

router = APIRouter()

MODELS_DIR = Path("/app/shared/models/ml")
CONFIG_FILE = MODELS_DIR / "parameters.json"


class ModelParameters(BaseModel):
    catboost: Optional[Dict[str, Any]] = None
    xgboost: Optional[Dict[str, Any]] = None
    lightgbm: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None


@router.get("/parameters")
async def get_parameters():
    """Get current model parameters"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    
    # Return defaults
    return {
        "catboost": {
            "learning_rate": 0.1,
            "depth": 6,
            "iterations": 100,
            "l2_leaf_reg": 3
        },
        "xgboost": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8
        },
        "lightgbm": {
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.8
        },
        "ensemble": {
            "type": "weighted_average",
            "weights": {
                "catboost": 0.33,
                "xgboost": 0.33,
                "lightgbm": 0.34
            }
        }
    }


@router.post("/parameters")
async def update_parameters(parameters: ModelParameters):
    """Update model parameters"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing or create new
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            current = json.load(f)
    else:
        current = {}
    
    # Update with new parameters
    if parameters.catboost:
        current["catboost"] = parameters.catboost
    if parameters.xgboost:
        current["xgboost"] = parameters.xgboost
    if parameters.lightgbm:
        current["lightgbm"] = parameters.lightgbm
    if parameters.ensemble:
        current["ensemble"] = parameters.ensemble
    
    # Save
    with open(CONFIG_FILE, "w") as f:
        json.dump(current, f, indent=2)
    
    return {
        "status": "updated",
        "parameters": current
    }


@router.get("/parameters/defaults")
async def get_default_parameters():
    """Get default parameter values"""
    return {
        "catboost": {
            "learning_rate": 0.1,
            "depth": 6,
            "iterations": 100,
            "l2_leaf_reg": 3
        },
        "xgboost": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "lightgbm": {
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8
        },
        "ensemble": {
            "type": "weighted_average",
            "weights": {
                "catboost": 0.33,
                "xgboost": 0.33,
                "lightgbm": 0.34
            }
        }
    }


@router.get("/comparison")
async def compare_models():
    """Compare model performance"""
    # TODO: Load model comparison metrics from training results
    return {
        "models": {
            "catboost": {
                "accuracy": 0.0,
                "f1": 0.0,
                "status": "not_trained"
            },
            "xgboost": {
                "accuracy": 0.0,
                "f1": 0.0,
                "status": "not_trained"
            },
            "lightgbm": {
                "accuracy": 0.0,
                "f1": 0.0,
                "status": "not_trained"
            },
            "ensemble": {
                "accuracy": 0.0,
                "f1": 0.0,
                "status": "not_trained"
            }
        }
    }

