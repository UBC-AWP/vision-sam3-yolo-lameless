import json
import os
from typing import List, Tuple

import joblib
import pandas as pd


def load_model_and_scaler(models_dir: str):
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    feature_info_path = os.path.join(models_dir, "feature_info.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file does not exist: {scaler_path}")
    if not os.path.exists(feature_info_path):
        raise FileNotFoundError(f"Feature info file does not exist: {feature_info_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    threshold = feature_info.get("threshold", 0.1)
    feature_columns = feature_info.get("feature_columns", [])

    return model, scaler, feature_columns, threshold


def prepare_features(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    available_features = [col for col in feature_columns if col in df.columns]

    X = df[available_features].copy()

    if "insufficient_detections" in df.columns:
        insufficient_mask = df["insufficient_detections"] == True
        if insufficient_mask.sum() > 0:
            for col in available_features:
                if col in X.columns:
                    median_val = X[~insufficient_mask][col].median() if (~insufficient_mask).sum() > 0 else 0
                    X.loc[insufficient_mask, col] = median_val if not pd.isna(median_val) else 0

    X = X.fillna(X.median())

    return X, available_features
