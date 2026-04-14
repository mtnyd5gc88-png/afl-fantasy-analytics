from __future__ import annotations

import json
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.features.engineering import (
    FEATURE_COLUMNS,
    FeatureConfig,
    build_feature_matrix,
    encode_positions,
    load_injury_exposure,
    load_raw_player_games,
)
from app.db.sync_session import sync_engine
from app.ml.boosted import fit_boosted


@dataclass
class TrainReport:
    baseline_rmse_holdout: float
    boosted_rmse_holdout: float
    tscv_boosted_mean_rmse: float
    holdout_cutoff_date: str
    n_train: int
    n_test: int
    boosted_backend: str


def _prepare_matrix(seasons: list[int] | None) -> tuple[pd.DataFrame, list[str]]:
    raw = load_raw_player_games(sync_engine, seasons)
    try:
        inj = load_injury_exposure(sync_engine)
    except Exception:
        inj = pd.DataFrame()
    feats = build_feature_matrix(raw, inj if not inj.empty else None, FeatureConfig())
    wide = encode_positions(feats)
    pos_cols = [c for c in wide.columns if c.startswith("pos_")]

    num_cols = [c for c in FEATURE_COLUMNS if c in wide.columns]
    use_cols = num_cols + pos_cols
    for c in use_cols:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(subset=use_cols + ["target_fp"])
    return wide, use_cols


def train_and_persist(seasons: list[int] | None = None) -> TrainReport:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    wide, use_cols = _prepare_matrix(seasons)
    wide = wide.sort_values(["game_date", "fixture_id", "player_id"])
    cut_idx = int(len(wide) * 0.8)
    train_df = wide.iloc[:cut_idx]
    test_df = wide.iloc[cut_idx:]
    holdout_date = str(test_df["game_date"].min().date())

    X_tr = np.nan_to_num(
        train_df[use_cols].to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    y_tr = train_df["target_fp"].to_numpy(dtype=float)
    X_te = np.nan_to_num(test_df[use_cols].to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    y_te = test_df["target_fp"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lr = Ridge(alpha=10.0)
    lr.fit(X_tr_s, y_tr)
    base_pred = lr.predict(X_te_s)
    base_rmse = float(np.sqrt(mean_squared_error(y_te, base_pred)))

    bm = fit_boosted(
        X_tr,
        y_tr,
        n_estimators=settings.xgb_n_estimators,
        max_depth=settings.xgb_max_depth,
        learning_rate=settings.xgb_learning_rate,
    )
    b_pred = bm.predict(X_te)
    b_rmse = float(np.sqrt(mean_squared_error(y_te, b_pred)))

    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores: list[float] = []
    X_all = np.nan_to_num(wide[use_cols].to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    y_all = wide["target_fp"].to_numpy(dtype=float)
    for tr_i, va_i in tscv.split(X_all):
        m = fit_boosted(
            X_all[tr_i],
            y_all[tr_i],
            n_estimators=min(120, settings.xgb_n_estimators),
            max_depth=settings.xgb_max_depth,
            learning_rate=settings.xgb_learning_rate,
        )
        p = m.predict(X_all[va_i])
        cv_scores.append(float(np.sqrt(mean_squared_error(y_all[va_i], p))))
    cv_mean = float(np.mean(cv_scores)) if cv_scores else float("nan")

    resid = y_te - b_pred
    sigma_global = float(np.std(resid))

    lr_path = settings.model_dir / "linear_baseline.joblib"
    meta_path = settings.model_dir / "metadata.json"

    bm.save(settings.model_dir, "boosted")
    joblib.dump(lr, lr_path)
    meta = {
        "feature_columns": use_cols,
        "calibration_sigma": sigma_global,
        "holdout_rmse_boosted": b_rmse,
        "holdout_rmse_linear": base_rmse,
        "tscv_mean_rmse": cv_mean,
        "holdout_cutoff_date": holdout_date,
        "boosted_kind": bm.kind,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return TrainReport(
        baseline_rmse_holdout=base_rmse,
        boosted_rmse_holdout=b_rmse,
        tscv_boosted_mean_rmse=cv_mean,
        holdout_cutoff_date=holdout_date,
        n_train=len(train_df),
        n_test=len(test_df),
        boosted_backend=bm.kind,
    )
