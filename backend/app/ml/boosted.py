"""Gradient boosting: prefers XGBoost when import/runtime works; else sklearn HGBR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class BoostedModel:
    kind: str  # "xgb" | "sklearn_hgbr"
    estimator: Any

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def save(self, model_dir: Path, base_name: str = "boosted") -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.kind == "xgb":
            self.estimator.save_model(model_dir / f"{base_name}_xgb.json")
        else:
            joblib.dump(self.estimator, model_dir / f"{base_name}_hgbr.joblib")

    @staticmethod
    def load(model_dir: Path, base_name: str, kind: str) -> "BoostedModel":
        if kind == "xgb":
            from xgboost import XGBRegressor

            m = XGBRegressor()
            m.load_model(model_dir / f"{base_name}_xgb.json")
            return BoostedModel("xgb", m)
        est = joblib.load(model_dir / f"{base_name}_hgbr.joblib")
        return BoostedModel("sklearn_hgbr", est)


def fit_boosted(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> BoostedModel:
    try:
        from xgboost import XGBRegressor

        est = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        est.fit(X_tr, y_tr)
        return BoostedModel("xgb", est)
    except Exception:
        pass
    est = HistGradientBoostingRegressor(
        max_iter=min(n_estimators, 400),
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_iter_no_change=20,
        early_stopping=True,
        validation_fraction=0.08,
    )
    est.fit(X_tr, y_tr)
    return BoostedModel("sklearn_hgbr", est)
