"""Train XGBoost + linear baseline and persist artifacts. Run from `backend/`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.train import train_and_persist  # noqa: E402


def main() -> None:
    rep = train_and_persist(seasons=None)
    print(
        f"Trained ({rep.boosted_backend}): holdout_linear_rmse={rep.baseline_rmse_holdout:.3f} "
        f"boosted_rmse={rep.boosted_rmse_holdout:.3f} tscv_mean={rep.tscv_boosted_mean_rmse:.3f} "
        f"cutoff={rep.holdout_cutoff_date} train={rep.n_train} test={rep.n_test}"
    )


if __name__ == "__main__":
    main()
