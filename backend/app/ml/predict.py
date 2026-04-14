from __future__ import annotations

import json
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.features.engineering import (
    FeatureConfig,
    build_feature_matrix,
    encode_positions,
    load_injury_exposure,
    load_raw_player_games,
)
from app.db.sync_session import sync_engine
from app.ml.boosted import BoostedModel


@dataclass
class PlayerProjection:
    player_id: int
    name: str
    team_id: int
    position: str
    price: float
    predicted_points: float
    ci_low: float
    ci_high: float
    variance_proxy: float
    value_vs_price: float


class PredictionService:
    """Loads persisted models and produces per-player projections from latest feature rows."""

    def __init__(self) -> None:
        self._boosted: BoostedModel | None = None
        self._lr = None
        self._features: list[str] = []
        self._sigma: float = 12.0
        self._reload()

    def _reload(self) -> None:
        meta_path = settings.model_dir / "metadata.json"
        lr_path = settings.model_dir / "linear_baseline.joblib"
        if not meta_path.exists():
            self._boosted = None
            self._lr = None
            self._features = []
            return
        meta = json.loads(meta_path.read_text())
        self._features = list(meta["feature_columns"])
        self._sigma = float(meta.get("calibration_sigma", 12.0))
        kind = meta.get("boosted_kind", "xgb")
        try:
            self._boosted = BoostedModel.load(settings.model_dir, "boosted", kind)
        except Exception:
            self._boosted = None
        self._lr = joblib.load(lr_path) if lr_path.exists() else None

    def is_ready(self) -> bool:
        return self._boosted is not None and bool(self._features)

    @property
    def sigma_global(self) -> float:
        return self._sigma

    def build_latest_feature_rows(self, seasons: list[int] | None = None) -> pd.DataFrame:
        raw = load_raw_player_games(sync_engine, seasons)
        try:
            inj = load_injury_exposure(sync_engine)
        except Exception:
            inj = pd.DataFrame()
        feats = build_feature_matrix(raw, inj if not inj.empty else None, FeatureConfig())
        wide = encode_positions(feats)
        for c in self._features:
            if c not in wide.columns:
                wide[c] = 0.0
        wide = wide.sort_values(["player_id", "game_date", "fixture_id"])
        latest = wide.groupby("player_id", as_index=False).tail(1)
        return latest

    def predict_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_ready():
            raise RuntimeError("Model artifacts missing. Run training script first.")
        X = frame[self._features].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pred = self._boosted.predict(X)
        return pred, X

    def interval(self, pred: float, fp_std: float | None) -> tuple[float, float]:
        blend = np.sqrt(self._sigma**2 + (0.35 * (fp_std or 0)) ** 2)
        z = 1.645
        return float(pred - z * blend), float(pred + z * blend)

    def project_all(self, seasons: list[int] | None = None) -> list[PlayerProjection]:
        latest = self.build_latest_feature_rows(seasons)
        pred, _ = self.predict_frame(latest)
        n_std = FeatureConfig().variance_window
        out: list[PlayerProjection] = []
        for i, row in latest.reset_index(drop=True).iterrows():
            p = float(pred[i])
            fp_std = float(row.get(f"fp_std_{n_std}", 0) or 0)
            lo, hi = self.interval(p, fp_std)
            price = float(row.get("current_price", 300_000))
            value = p / max(price / 100_000.0, 1e-6)
            out.append(
                PlayerProjection(
                    player_id=int(row["player_id"]),
                    name=str(row["player_name"]),
                    team_id=int(row["player_team_id"]),
                    position=str(row["primary_position"]),
                    price=price,
                    predicted_points=p,
                    ci_low=lo,
                    ci_high=hi,
                    variance_proxy=fp_std,
                    value_vs_price=float(value),
                )
            )
        return out
