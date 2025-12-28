"""
ML Configuration Router
Provides endpoints for configuring CatBoost, XGBoost, LightGBM parameters
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import yaml

router = APIRouter()

# Configuration paths
CONFIG_DIR = Path("/app/data/training/ml_config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("/app/shared/models/ml")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============== Pydantic Models ==============

class CatBoostConfig(BaseModel):
    """CatBoost model configuration"""
    iterations: int = Field(default=100, ge=10, le=10000, description="Number of boosting iterations")
    learning_rate: float = Field(default=0.1, ge=0.001, le=1.0, description="Learning rate")
    depth: int = Field(default=6, ge=1, le=16, description="Maximum tree depth")
    l2_leaf_reg: float = Field(default=3.0, ge=0.0, le=100.0, description="L2 regularization coefficient")
    random_strength: float = Field(default=1.0, ge=0.0, le=10.0, description="Random strength for scoring")
    bagging_temperature: float = Field(default=1.0, ge=0.0, le=10.0, description="Bayesian bootstrap temperature")
    border_count: int = Field(default=254, ge=1, le=255, description="Number of splits for numerical features")
    grow_policy: str = Field(default="SymmetricTree", description="Tree growing policy: SymmetricTree, Depthwise, Lossguide")
    bootstrap_type: str = Field(default="MVS", description="Bootstrap type: Bayesian, Bernoulli, MVS, No")
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    class Config:
        json_schema_extra = {
            "example": {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "l2_leaf_reg": 3.0
            }
        }


class XGBoostConfig(BaseModel):
    """XGBoost model configuration"""
    n_estimators: int = Field(default=100, ge=10, le=10000, description="Number of boosting rounds")
    learning_rate: float = Field(default=0.1, ge=0.001, le=1.0, description="Learning rate (eta)")
    max_depth: int = Field(default=6, ge=1, le=20, description="Maximum tree depth")
    min_child_weight: float = Field(default=1.0, ge=0.0, le=100.0, description="Minimum sum of instance weight in child")
    gamma: float = Field(default=0.0, ge=0.0, le=10.0, description="Min loss reduction for split")
    subsample: float = Field(default=1.0, ge=0.1, le=1.0, description="Subsample ratio of training data")
    colsample_bytree: float = Field(default=1.0, ge=0.1, le=1.0, description="Subsample ratio of columns")
    colsample_bylevel: float = Field(default=1.0, ge=0.1, le=1.0, description="Subsample ratio per level")
    reg_alpha: float = Field(default=0.0, ge=0.0, le=100.0, description="L1 regularization")
    reg_lambda: float = Field(default=1.0, ge=0.0, le=100.0, description="L2 regularization")
    scale_pos_weight: float = Field(default=1.0, ge=0.1, le=100.0, description="Balance of positive/negative weights")
    booster: str = Field(default="gbtree", description="Booster type: gbtree, gblinear, dart")
    tree_method: str = Field(default="hist", description="Tree method: auto, exact, approx, hist, gpu_hist")
    random_state: int = Field(default=42, ge=0, description="Random seed")

    class Config:
        json_schema_extra = {
            "example": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8
            }
        }


class LightGBMConfig(BaseModel):
    """LightGBM model configuration"""
    n_estimators: int = Field(default=100, ge=10, le=10000, description="Number of boosting iterations")
    learning_rate: float = Field(default=0.1, ge=0.001, le=1.0, description="Learning rate")
    max_depth: int = Field(default=6, ge=-1, le=20, description="Maximum tree depth (-1 for no limit)")
    num_leaves: int = Field(default=31, ge=2, le=131072, description="Maximum number of leaves")
    min_child_samples: int = Field(default=20, ge=1, le=1000, description="Minimum samples in leaf")
    min_child_weight: float = Field(default=0.001, ge=0.0, le=100.0, description="Minimum sum of hessian in leaf")
    subsample: float = Field(default=1.0, ge=0.1, le=1.0, description="Subsample ratio of training data")
    colsample_bytree: float = Field(default=1.0, ge=0.1, le=1.0, description="Subsample ratio of columns")
    reg_alpha: float = Field(default=0.0, ge=0.0, le=100.0, description="L1 regularization")
    reg_lambda: float = Field(default=0.0, ge=0.0, le=100.0, description="L2 regularization")
    min_split_gain: float = Field(default=0.0, ge=0.0, le=10.0, description="Minimum gain to make a split")
    boosting_type: str = Field(default="gbdt", description="Boosting type: gbdt, dart, goss, rf")
    objective: str = Field(default="binary", description="Objective function")
    random_state: int = Field(default=42, ge=0, description="Random seed")

    class Config:
        json_schema_extra = {
            "example": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31
            }
        }


class EnsembleConfig(BaseModel):
    """Ensemble configuration for combining model predictions"""
    catboost_weight: float = Field(default=0.33, ge=0.0, le=1.0, description="Weight for CatBoost")
    xgboost_weight: float = Field(default=0.33, ge=0.0, le=1.0, description="Weight for XGBoost")
    lightgbm_weight: float = Field(default=0.34, ge=0.0, le=1.0, description="Weight for LightGBM")
    voting_method: str = Field(default="soft", description="Voting method: soft (probability), hard (class)")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Classification threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "catboost_weight": 0.4,
                "xgboost_weight": 0.3,
                "lightgbm_weight": 0.3,
                "voting_method": "soft",
                "threshold": 0.5
            }
        }


class TrainingConfig(BaseModel):
    """Training configuration"""
    min_samples: int = Field(default=10, ge=2, le=1000, description="Minimum samples to start training")
    cv_folds: int = Field(default=5, ge=2, le=20, description="Number of cross-validation folds")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set ratio")
    stratify: bool = Field(default=True, description="Use stratified splits")
    shuffle: bool = Field(default=True, description="Shuffle data before splitting")
    early_stopping_rounds: Optional[int] = Field(default=None, ge=1, le=100, description="Early stopping rounds (optional)")
    feature_selection: bool = Field(default=False, description="Enable feature selection")
    scale_features: bool = Field(default=True, description="Standardize features before training")


class FullMLConfig(BaseModel):
    """Complete ML configuration"""
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# ============== Helper Functions ==============

def get_config_file() -> Path:
    return CONFIG_DIR / "ml_config.json"


def load_config() -> Dict[str, Any]:
    """Load ML configuration from file"""
    config_file = get_config_file()
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    # Return default config
    return FullMLConfig().model_dump()


def save_config(config: Dict[str, Any]) -> None:
    """Save ML configuration to file"""
    config_file = get_config_file()
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def save_ensemble_weights(weights: Dict[str, float]) -> None:
    """Save ensemble weights for runtime use"""
    weights_file = MODELS_DIR / "ensemble_weights.json"
    with open(weights_file, "w") as f:
        json.dump(weights, f, indent=2)


# ============== API Endpoints ==============

@router.get("/")
async def get_ml_config():
    """Get complete ML configuration"""
    config = load_config()
    return {
        "config": config,
        "config_file": str(get_config_file()),
        "models_dir": str(MODELS_DIR)
    }


@router.put("/")
async def update_ml_config(config: FullMLConfig):
    """Update complete ML configuration"""
    config_dict = config.model_dump()
    save_config(config_dict)

    # Also save ensemble weights for runtime
    ensemble = config.ensemble
    save_ensemble_weights({
        "catboost": ensemble.catboost_weight,
        "xgboost": ensemble.xgboost_weight,
        "lightgbm": ensemble.lightgbm_weight
    })

    return {
        "message": "Configuration updated successfully",
        "config": config_dict
    }


@router.get("/catboost")
async def get_catboost_config():
    """Get CatBoost configuration"""
    config = load_config()
    return {
        "config": config.get("catboost", CatBoostConfig().model_dump()),
        "schema": CatBoostConfig.model_json_schema()
    }


@router.put("/catboost")
async def update_catboost_config(catboost_config: CatBoostConfig):
    """Update CatBoost configuration"""
    config = load_config()
    config["catboost"] = catboost_config.model_dump()
    save_config(config)
    return {
        "message": "CatBoost configuration updated",
        "config": config["catboost"]
    }


@router.get("/xgboost")
async def get_xgboost_config():
    """Get XGBoost configuration"""
    config = load_config()
    return {
        "config": config.get("xgboost", XGBoostConfig().model_dump()),
        "schema": XGBoostConfig.model_json_schema()
    }


@router.put("/xgboost")
async def update_xgboost_config(xgboost_config: XGBoostConfig):
    """Update XGBoost configuration"""
    config = load_config()
    config["xgboost"] = xgboost_config.model_dump()
    save_config(config)
    return {
        "message": "XGBoost configuration updated",
        "config": config["xgboost"]
    }


@router.get("/lightgbm")
async def get_lightgbm_config():
    """Get LightGBM configuration"""
    config = load_config()
    return {
        "config": config.get("lightgbm", LightGBMConfig().model_dump()),
        "schema": LightGBMConfig.model_json_schema()
    }


@router.put("/lightgbm")
async def update_lightgbm_config(lightgbm_config: LightGBMConfig):
    """Update LightGBM configuration"""
    config = load_config()
    config["lightgbm"] = lightgbm_config.model_dump()
    save_config(config)
    return {
        "message": "LightGBM configuration updated",
        "config": config["lightgbm"]
    }


@router.get("/ensemble")
async def get_ensemble_config():
    """Get ensemble configuration"""
    config = load_config()
    return {
        "config": config.get("ensemble", EnsembleConfig().model_dump()),
        "schema": EnsembleConfig.model_json_schema()
    }


@router.put("/ensemble")
async def update_ensemble_config(ensemble_config: EnsembleConfig):
    """Update ensemble configuration"""
    config = load_config()
    config["ensemble"] = ensemble_config.model_dump()
    save_config(config)

    # Update runtime weights
    save_ensemble_weights({
        "catboost": ensemble_config.catboost_weight,
        "xgboost": ensemble_config.xgboost_weight,
        "lightgbm": ensemble_config.lightgbm_weight
    })

    return {
        "message": "Ensemble configuration updated",
        "config": config["ensemble"]
    }


@router.get("/training")
async def get_training_config():
    """Get training configuration"""
    config = load_config()
    return {
        "config": config.get("training", TrainingConfig().model_dump()),
        "schema": TrainingConfig.model_json_schema()
    }


@router.put("/training")
async def update_training_config(training_config: TrainingConfig):
    """Update training configuration"""
    config = load_config()
    config["training"] = training_config.model_dump()
    save_config(config)
    return {
        "message": "Training configuration updated",
        "config": config["training"]
    }


@router.post("/reset")
async def reset_to_defaults():
    """Reset all configurations to defaults"""
    default_config = FullMLConfig()
    config_dict = default_config.model_dump()
    save_config(config_dict)

    # Reset ensemble weights
    save_ensemble_weights({
        "catboost": 0.33,
        "xgboost": 0.33,
        "lightgbm": 0.34
    })

    return {
        "message": "Configuration reset to defaults",
        "config": config_dict
    }


@router.get("/schema")
async def get_config_schema():
    """Get JSON schema for all configurations"""
    return {
        "catboost": CatBoostConfig.model_json_schema(),
        "xgboost": XGBoostConfig.model_json_schema(),
        "lightgbm": LightGBMConfig.model_json_schema(),
        "ensemble": EnsembleConfig.model_json_schema(),
        "training": TrainingConfig.model_json_schema(),
        "full": FullMLConfig.model_json_schema()
    }


@router.get("/models/status")
async def get_models_status():
    """Get status of trained models"""
    models_status = {
        "catboost": {
            "trained": False,
            "file": None,
            "size": None
        },
        "xgboost": {
            "trained": False,
            "file": None,
            "size": None
        },
        "lightgbm": {
            "trained": False,
            "file": None,
            "size": None
        },
        "ensemble": {
            "trained": False,
            "file": None,
            "size": None
        }
    }

    # Check CatBoost
    catboost_file = MODELS_DIR / "catboost_latest.cbm"
    if catboost_file.exists():
        models_status["catboost"] = {
            "trained": True,
            "file": str(catboost_file),
            "size": catboost_file.stat().st_size
        }

    # Check XGBoost
    xgboost_file = MODELS_DIR / "xgboost_latest.json"
    if xgboost_file.exists():
        models_status["xgboost"] = {
            "trained": True,
            "file": str(xgboost_file),
            "size": xgboost_file.stat().st_size
        }

    # Check LightGBM
    lightgbm_file = MODELS_DIR / "lightgbm_latest.txt"
    if lightgbm_file.exists():
        models_status["lightgbm"] = {
            "trained": True,
            "file": str(lightgbm_file),
            "size": lightgbm_file.stat().st_size
        }

    # Check ensemble weights
    weights_file = MODELS_DIR / "ensemble_weights.json"
    if weights_file.exists():
        with open(weights_file) as f:
            weights = json.load(f)
        models_status["ensemble"] = {
            "trained": True,
            "file": str(weights_file),
            "weights": weights
        }

    # Check training status
    training_status_file = Path("/app/data/training/training_status.json")
    training_status = None
    if training_status_file.exists():
        with open(training_status_file) as f:
            training_status = json.load(f)

    return {
        "models": models_status,
        "training_status": training_status,
        "models_dir": str(MODELS_DIR)
    }


@router.get("/parameter-descriptions")
async def get_parameter_descriptions():
    """Get detailed descriptions for all ML parameters"""
    return {
        "catboost": {
            "iterations": {
                "name": "Iterations",
                "description": "Number of boosting iterations (trees to build). Higher values can improve accuracy but increase training time and risk of overfitting.",
                "category": "Training",
                "default": 100,
                "range": [10, 10000]
            },
            "learning_rate": {
                "name": "Learning Rate",
                "description": "Step size for gradient descent. Lower values require more iterations but can achieve better accuracy.",
                "category": "Training",
                "default": 0.1,
                "range": [0.001, 1.0]
            },
            "depth": {
                "name": "Tree Depth",
                "description": "Maximum depth of trees. Deeper trees can capture more complex patterns but may overfit.",
                "category": "Tree Structure",
                "default": 6,
                "range": [1, 16]
            },
            "l2_leaf_reg": {
                "name": "L2 Regularization",
                "description": "Coefficient for L2 regularization. Higher values prevent overfitting but may underfit.",
                "category": "Regularization",
                "default": 3.0,
                "range": [0.0, 100.0]
            },
            "random_strength": {
                "name": "Random Strength",
                "description": "Amount of randomness for scoring splits. Higher values add more randomization.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.0, 10.0]
            },
            "bagging_temperature": {
                "name": "Bagging Temperature",
                "description": "Controls intensity of Bayesian bootstrap. Higher values increase randomization.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.0, 10.0]
            },
            "border_count": {
                "name": "Border Count",
                "description": "Number of splits for numerical features. Higher values give more precision but slower training.",
                "category": "Tree Structure",
                "default": 254,
                "range": [1, 255]
            },
            "grow_policy": {
                "name": "Grow Policy",
                "description": "Tree growing policy. SymmetricTree (default, fastest), Depthwise (depth-first), Lossguide (best-first).",
                "category": "Tree Structure",
                "default": "SymmetricTree",
                "options": ["SymmetricTree", "Depthwise", "Lossguide"]
            },
            "bootstrap_type": {
                "name": "Bootstrap Type",
                "description": "Bootstrap sampling method. MVS (recommended), Bayesian, Bernoulli, or No bootstrap.",
                "category": "Regularization",
                "default": "MVS",
                "options": ["Bayesian", "Bernoulli", "MVS", "No"]
            }
        },
        "xgboost": {
            "n_estimators": {
                "name": "Number of Estimators",
                "description": "Number of boosting rounds. More rounds can improve accuracy but increase training time.",
                "category": "Training",
                "default": 100,
                "range": [10, 10000]
            },
            "learning_rate": {
                "name": "Learning Rate (eta)",
                "description": "Step size shrinkage to prevent overfitting. Lower values require more boosting rounds.",
                "category": "Training",
                "default": 0.1,
                "range": [0.001, 1.0]
            },
            "max_depth": {
                "name": "Maximum Depth",
                "description": "Maximum depth of a tree. Deeper trees can capture complex patterns but may overfit.",
                "category": "Tree Structure",
                "default": 6,
                "range": [1, 20]
            },
            "min_child_weight": {
                "name": "Minimum Child Weight",
                "description": "Minimum sum of instance weight needed in a child. Higher values prevent overfitting.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.0, 100.0]
            },
            "gamma": {
                "name": "Gamma (min_split_loss)",
                "description": "Minimum loss reduction required to make a split. Higher values = more conservative.",
                "category": "Regularization",
                "default": 0.0,
                "range": [0.0, 10.0]
            },
            "subsample": {
                "name": "Subsample Ratio",
                "description": "Fraction of samples used for training each tree. Lower values prevent overfitting.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.1, 1.0]
            },
            "colsample_bytree": {
                "name": "Column Sample by Tree",
                "description": "Fraction of features used for each tree. Adds randomness and prevents overfitting.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.1, 1.0]
            },
            "reg_alpha": {
                "name": "L1 Regularization (alpha)",
                "description": "L1 regularization on weights. Helps with feature selection.",
                "category": "Regularization",
                "default": 0.0,
                "range": [0.0, 100.0]
            },
            "reg_lambda": {
                "name": "L2 Regularization (lambda)",
                "description": "L2 regularization on weights. Prevents overfitting.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.0, 100.0]
            },
            "scale_pos_weight": {
                "name": "Scale Positive Weight",
                "description": "Balance of positive/negative weights. Use ratio of negative/positive for imbalanced data.",
                "category": "Training",
                "default": 1.0,
                "range": [0.1, 100.0]
            },
            "booster": {
                "name": "Booster Type",
                "description": "Booster to use. gbtree (tree-based), gblinear (linear), dart (dropout).",
                "category": "Model",
                "default": "gbtree",
                "options": ["gbtree", "gblinear", "dart"]
            },
            "tree_method": {
                "name": "Tree Method",
                "description": "Tree construction algorithm. hist is fastest, exact is most accurate.",
                "category": "Model",
                "default": "hist",
                "options": ["auto", "exact", "approx", "hist", "gpu_hist"]
            }
        },
        "lightgbm": {
            "n_estimators": {
                "name": "Number of Estimators",
                "description": "Number of boosting iterations. More iterations can improve accuracy.",
                "category": "Training",
                "default": 100,
                "range": [10, 10000]
            },
            "learning_rate": {
                "name": "Learning Rate",
                "description": "Boosting learning rate. Smaller values require more iterations.",
                "category": "Training",
                "default": 0.1,
                "range": [0.001, 1.0]
            },
            "max_depth": {
                "name": "Maximum Depth",
                "description": "Maximum depth of trees. -1 means no limit. Deeper trees = more complex patterns.",
                "category": "Tree Structure",
                "default": 6,
                "range": [-1, 20]
            },
            "num_leaves": {
                "name": "Number of Leaves",
                "description": "Maximum number of leaves in one tree. Main parameter for tree complexity.",
                "category": "Tree Structure",
                "default": 31,
                "range": [2, 131072]
            },
            "min_child_samples": {
                "name": "Minimum Child Samples",
                "description": "Minimum number of data points in a leaf. Higher values prevent overfitting.",
                "category": "Regularization",
                "default": 20,
                "range": [1, 1000]
            },
            "subsample": {
                "name": "Subsample (bagging_fraction)",
                "description": "Fraction of data used for training each iteration. Adds randomness.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.1, 1.0]
            },
            "colsample_bytree": {
                "name": "Column Sample (feature_fraction)",
                "description": "Fraction of features used for each tree. Prevents overfitting.",
                "category": "Regularization",
                "default": 1.0,
                "range": [0.1, 1.0]
            },
            "reg_alpha": {
                "name": "L1 Regularization",
                "description": "L1 regularization term. Helps with feature selection.",
                "category": "Regularization",
                "default": 0.0,
                "range": [0.0, 100.0]
            },
            "reg_lambda": {
                "name": "L2 Regularization",
                "description": "L2 regularization term. Prevents overfitting.",
                "category": "Regularization",
                "default": 0.0,
                "range": [0.0, 100.0]
            },
            "min_split_gain": {
                "name": "Minimum Split Gain",
                "description": "Minimum gain to make a split. Higher values = more conservative.",
                "category": "Regularization",
                "default": 0.0,
                "range": [0.0, 10.0]
            },
            "boosting_type": {
                "name": "Boosting Type",
                "description": "Boosting algorithm. gbdt (default), dart (dropout), goss (gradient-based), rf (random forest).",
                "category": "Model",
                "default": "gbdt",
                "options": ["gbdt", "dart", "goss", "rf"]
            }
        },
        "ensemble": {
            "catboost_weight": {
                "name": "CatBoost Weight",
                "description": "Weight for CatBoost model in ensemble voting. Higher weight = more influence.",
                "category": "Ensemble",
                "default": 0.33,
                "range": [0.0, 1.0]
            },
            "xgboost_weight": {
                "name": "XGBoost Weight",
                "description": "Weight for XGBoost model in ensemble voting.",
                "category": "Ensemble",
                "default": 0.33,
                "range": [0.0, 1.0]
            },
            "lightgbm_weight": {
                "name": "LightGBM Weight",
                "description": "Weight for LightGBM model in ensemble voting.",
                "category": "Ensemble",
                "default": 0.34,
                "range": [0.0, 1.0]
            },
            "voting_method": {
                "name": "Voting Method",
                "description": "soft = weighted average of probabilities, hard = majority voting of predictions.",
                "category": "Ensemble",
                "default": "soft",
                "options": ["soft", "hard"]
            },
            "threshold": {
                "name": "Classification Threshold",
                "description": "Probability threshold for positive class. Lower = more sensitive, Higher = more specific.",
                "category": "Ensemble",
                "default": 0.5,
                "range": [0.0, 1.0]
            }
        }
    }
