"""
Trade optimization: mixed-integer program on candidate swaps + Monte Carlo EV.

Deterministic EV uses horizon * (mean_in - mean_out). Risk uses std of simulated
team point deltas. Candidates are filtered by budget and position; conflicts
(one player involved in multiple chosen trades) are forbidden via constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from app.ml.predict import PlayerProjection, PredictionService


@dataclass(frozen=True)
class TradeRecommendation:
    trade_out_id: int
    trade_in_id: int
    expected_value_gain: float
    risk_team_delta_std: float
    net_budget_delta: float
    mc_mean: float
    mc_std: float


class TradeOptimizer:
    def __init__(self, svc: PredictionService) -> None:
        self.svc = svc

    def _simulate_swap_gain(
        self,
        lineup_means: np.ndarray,
        lineup_sigs: np.ndarray,
        out_idx: int,
        in_mean: float,
        in_sig: float,
        horizon: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Per-round sums are i.i.d. normal approximations; horizon scales mean/variance."""
        base = np.zeros(n_samples)
        prop = np.zeros(n_samples)
        for _ in range(horizon):
            draws = rng.normal(lineup_means, lineup_sigs, size=(n_samples, len(lineup_means)))
            base += draws.sum(axis=1)
            m = lineup_means.copy()
            s = lineup_sigs.copy()
            m[out_idx] = in_mean
            s[out_idx] = in_sig
            draws2 = rng.normal(m, s, size=(n_samples, len(lineup_means)))
            prop += draws2.sum(axis=1)
        delta = prop - base
        return float(delta.mean()), float(delta.std(ddof=1)) if n_samples > 1 else 0.0

    def recommend(
        self,
        field_player_ids: list[int],
        bank: float,
        max_trades: int,
        horizon_rounds: int,
        mc_samples: int,
        rng: np.random.Generator | None = None,
    ) -> list[TradeRecommendation]:
        rng = rng or np.random.default_rng(42)
        if not self.svc.is_ready():
            raise RuntimeError("Model not trained")
        projs = self.svc.project_all()
        by_id = {p.player_id: p for p in projs}
        field = [pid for pid in field_player_ids if pid in by_id]
        if len(field) < 3:
            return []

        lineup_means = np.array([by_id[pid].predicted_points for pid in field], dtype=float)
        lineup_sigs = np.array(
            [max(by_id[pid].variance_proxy, 1.0) * 0.85 + self.svc.sigma_global * 0.35 for pid in field],
            dtype=float,
        )

        team_pos = {pid: by_id[pid].position for pid in field}
        candidates: list[dict] = []

        for idx_out, pid_out in enumerate(field):
            pos = team_pos[pid_out]
            p_out = by_id[pid_out]
            for cand in projs:
                if cand.player_id in field:
                    continue
                if cand.position != pos:
                    continue
                net_price = cand.price - p_out.price
                if net_price > bank + 1e-6:
                    continue
                mc_mean, mc_std = self._simulate_swap_gain(
                    lineup_means,
                    lineup_sigs,
                    idx_out,
                    cand.predicted_points,
                    max(cand.variance_proxy, 1.0),
                    horizon_rounds,
                    mc_samples,
                    rng,
                )
                det_ev = horizon_rounds * (cand.predicted_points - p_out.predicted_points)
                candidates.append(
                    {
                        "out": pid_out,
                        "in": cand.player_id,
                        "ev": det_ev,
                        "mc_mean": mc_mean,
                        "mc_std": mc_std,
                        "net_price": net_price,
                        "idx_out": idx_out,
                    }
                )

        if not candidates:
            return []

        candidates.sort(key=lambda c: c["mc_mean"], reverse=True)
        candidates = candidates[: min(len(candidates), 96)]

        prob = pulp.LpProblem("afl_trades", pulp.LpMaximize)
        z_vars = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(len(candidates))]

        prob += pulp.lpSum(z_vars[i] * candidates[i]["ev"] for i in range(len(candidates)))

        prob += pulp.lpSum(z_vars) <= max_trades
        prob += pulp.lpSum(z_vars[i] * candidates[i]["net_price"] for i in range(len(candidates))) <= bank

        player_usage: dict[int, list[int]] = {}
        for i, c in enumerate(candidates):
            player_usage.setdefault(c["out"], []).append(i)
            player_usage.setdefault(c["in"], []).append(i)
        for _, idxs in player_usage.items():
            if len(idxs) > 1:
                prob += pulp.lpSum(z_vars[j] for j in idxs) <= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]
        chosen: list[int] = []
        if status in ("Optimal", "Feasible"):
            chosen = [i for i, v in enumerate(z_vars) if (v.value() or 0) > 0.5]

        if not chosen:
            used: set[int] = set()
            budget_left = float(bank)
            trades_left = int(max_trades)
            for i, c in enumerate(candidates):
                if trades_left <= 0:
                    break
                if c["net_price"] > budget_left + 1e-6:
                    continue
                if c["out"] in used or c["in"] in used:
                    continue
                chosen.append(i)
                used.add(c["out"])
                used.add(c["in"])
                budget_left -= c["net_price"]
                trades_left -= 1

        recs: list[TradeRecommendation] = []
        for i in chosen:
            c = candidates[i]
            recs.append(
                TradeRecommendation(
                    trade_out_id=int(c["out"]),
                    trade_in_id=int(c["in"]),
                    expected_value_gain=float(c["ev"]),
                    risk_team_delta_std=float(c["mc_std"]),
                    net_budget_delta=float(c["net_price"]),
                    mc_mean=float(c["mc_mean"]),
                    mc_std=float(c["mc_std"]),
                )
            )
        recs.sort(key=lambda r: r.mc_mean, reverse=True)
        return recs
