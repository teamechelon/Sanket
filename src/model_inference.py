"""Persisted Phase 32 XGBoost pipeline loading and strict risk-score inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.schedule_robustness import FEATURES


ARTIFACT_KIND = "PHASE_32_GENERATED_XGBOOST_PIPELINE"


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize_pipeline(model, path: Path) -> dict[str, object]:
    """Persist the fitted sklearn Pipeline without altering its estimator state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=0, protocol=5)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_path": str(path),
        "sha256": artifact_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load_pipeline(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"model artifact does not exist: {path}")
    return joblib.load(path)


def validate_feature_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("inference input must be a pandas DataFrame")
    actual = list(frame.columns)
    if actual != FEATURES:
        missing = [name for name in FEATURES if name not in actual]
        unexpected = [name for name in actual if name not in FEATURES]
        if missing or unexpected:
            raise ValueError(f"incompatible feature schema; missing={missing}; unexpected={unexpected}")
        raise ValueError("incompatible feature schema; canonical feature ordering is required")


def predict_risk_scores(model, frame: pd.DataFrame) -> np.ndarray:
    """Return uncalibrated relative-risk ranking scores; no threshold is applied."""
    validate_feature_frame(frame)
    scores = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("model returned invalid risk scores")
    return scores
