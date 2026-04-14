"""
Feature engineering for AFL Fantasy projection models.

All features are computed from historical rows only (shifted / expanding) to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    rolling_windows: tuple[int, ...] = (3, 5)
    ew_span: int = 5
    variance_window: int = 5
    role_shift_z_threshold: float = 2.0


def _sync_sqlite_url(async_url: str) -> str:
    if "+aiosqlite" in async_url:
        rest = async_url.split("///", 1)[-1]
        return "sqlite:///" + rest
    return async_url


def load_raw_player_games(engine, seasons: list[int] | None = None) -> pd.DataFrame:
    q = """
    SELECT
      pgs.id AS pgs_id,
      p.id AS player_id,
      p.name AS player_name,
      p.primary_position,
      p.team_id AS player_team_id,
      p.current_price,
      pgs.fantasy_points,
      pgs.disposals, pgs.kicks, pgs.handballs, pgs.marks, pgs.tackles,
      pgs.hitouts, pgs.goals, pgs.behinds, pgs.clearances,
      pgs.contested_possessions, pgs.uncontested_possessions,
      pgs.inside_50s, pgs.rebound_50s, pgs.clangers,
      pgs.frees_for, pgs.frees_against, pgs.time_on_ground_pct,
      pgs.is_home,
      pgs.opponent_team_id,
      pgs.team_id AS stat_team_id,
      f.id AS fixture_id,
      f.season, f.round_number, f.game_date,
      f.expected_total_points,
      tgs.possessions_share AS team_possession_share,
      tgs.disposals AS team_disposals,
      tgs.contested_possessions AS team_contested,
      tgs.inside_50s AS team_inside_50
    FROM player_game_stats pgs
    JOIN players p ON p.id = pgs.player_id
    JOIN fixtures f ON f.id = pgs.fixture_id
    LEFT JOIN team_game_stats tgs
      ON tgs.fixture_id = pgs.fixture_id AND tgs.team_id = pgs.team_id
    """
    if seasons:
        q += " WHERE f.season IN (" + ",".join(str(s) for s in seasons) + ")"
    q += " ORDER BY f.game_date, f.id, p.id"
    return pd.read_sql_query(q, engine)


def load_injury_exposure(engine) -> pd.DataFrame:
    """Per player: latest signal impact (0-1) and binary OUT."""
    q = """
    SELECT player_id, status, impact_score, recorded_at
    FROM player_availability_signals
    ORDER BY player_id, recorded_at
    """
    return pd.read_sql_query(q, engine)


def build_feature_matrix(raw: pd.DataFrame, inj: pd.DataFrame | None, cfg: FeatureConfig) -> pd.DataFrame:
    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date", "fixture_id"]).reset_index(drop=True)

    g = df.groupby("player_id", group_keys=False)

    # --- Form: rolling means & exponential weighted ---
    for w in cfg.rolling_windows:
        df[f"roll_fp_mean_{w}"] = g["fantasy_points"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"roll_disp_mean_{w}"] = g["disposals"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    df["ew_fp"] = g["fantasy_points"].transform(
        lambda s: s.shift(1).ewm(span=cfg.ew_span, adjust=False).mean()
    )

    # --- Usage / role ---
    df["tog_pct"] = df["time_on_ground_pct"]
    df["_disp_share_row"] = df["disposals"] / df["team_disposals"].replace(0, np.nan)
    df["disp_share_roll5"] = g["_disp_share_row"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["disp_share_roll5"] = df["disp_share_roll5"].fillna(df["_disp_share_row"])
    df.drop(columns=["_disp_share_row"], inplace=True)

    # Position proxy from stat mix (complements roster position)
    z = df["hitouts"].clip(lower=0) + 1
    df["proxy_ruckish"] = df["hitouts"] / z
    df["proxy_defensive"] = (df["rebound_50s"] + df["marks"]) / (df["disposals"] + 1)
    df["proxy_inside"] = df["inside_50s"] / (df["disposals"] + 1)

    prev_disp = g["disposals"].shift(1)
    roll_mu = g["disposals"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    roll_sd = g["disposals"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).std()).replace(0, np.nan)
    df["role_shift_z"] = ((prev_disp - roll_mu) / roll_sd).fillna(0.0)
    df["role_change_flag"] = (df["role_shift_z"].abs() >= cfg.role_shift_z_threshold).astype(float)

    # --- Match context ---
    df["is_home"] = df["is_home"].astype(float)
    df["expected_total"] = df["expected_total_points"].fillna(df["expected_total_points"].median())

    df = df.sort_values(["game_date", "fixture_id"])
    df["opp_def_vs_pos"] = (
        df.groupby(["opponent_team_id", "primary_position"])["fantasy_points"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(df.groupby("primary_position")["fantasy_points"].transform("mean"))
    )

    opp_disp = (
        df.groupby(["fixture_id", "stat_team_id"], as_index=False)
        .agg(opp_team_disposals=("team_disposals", "first"))
        .rename(columns={"stat_team_id": "opp_team_key"})
    )
    df = df.merge(
        opp_disp,
        left_on=["fixture_id", "opponent_team_id"],
        right_on=["fixture_id", "opp_team_key"],
        how="left",
    )
    df = df.drop(columns=["opp_team_key"], errors="ignore")
    df = df.sort_values(["game_date", "fixture_id"])
    df["opp_pace_roll5"] = df.groupby("opponent_team_id")["opp_team_disposals"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["opp_pace_roll5"] = df["opp_pace_roll5"].fillna(df["opp_team_disposals"])

    # --- Team context ---
    df["team_possession_share"] = df["team_possession_share"].fillna(0.5)

    if inj is not None and not inj.empty:
        inj = inj.copy()
        inj["recorded_at"] = pd.to_datetime(inj["recorded_at"])
        latest = inj.sort_values("recorded_at").groupby("player_id", as_index=False).tail(1)
        roster = df[["player_id", "player_team_id"]].drop_duplicates()
        latest = latest.merge(roster, on="player_id", how="left")
        outs = latest.loc[latest["status"] == "OUT"]
        out_by_team = outs.groupby("player_team_id")["player_id"].count()
        df["teammate_out_count"] = df["player_team_id"].map(out_by_team).fillna(0.0).astype(float)
        self_out_ids = set(latest.loc[latest["status"] == "OUT", "player_id"].astype(int))
        df["teammate_out_count"] = df["teammate_out_count"] - df["player_id"].isin(self_out_ids).astype(float)
        df["teammate_out_count"] = df["teammate_out_count"].clip(lower=0.0)
    else:
        df["teammate_out_count"] = 0.0

    df = df.sort_values(["player_id", "game_date", "fixture_id"])
    g2 = df.groupby("player_id", group_keys=False)
    n = cfg.variance_window
    df[f"fp_std_{n}"] = g2["fantasy_points"].transform(
        lambda s: s.shift(1).rolling(n, min_periods=min(2, n)).std()
    )
    df[f"fp_ceiling_{n}"] = g2["fantasy_points"].transform(
        lambda s: s.shift(1).rolling(n, min_periods=1).max()
    )
    df[f"fp_floor_{n}"] = g2["fantasy_points"].transform(
        lambda s: s.shift(1).rolling(n, min_periods=1).min()
    )

    df["target_fp"] = df["fantasy_points"]

    return df


FEATURE_COLUMNS = [
    "roll_fp_mean_3",
    "roll_fp_mean_5",
    "roll_disp_mean_3",
    "roll_disp_mean_5",
    "ew_fp",
    "tog_pct",
    "disp_share_roll5",
    "proxy_ruckish",
    "proxy_defensive",
    "proxy_inside",
    "role_change_flag",
    "role_shift_z",
    "is_home",
    "expected_total",
    "opp_def_vs_pos",
    "opp_pace_roll5",
    "team_possession_share",
    "teammate_out_count",
    "fp_std_5",
    "fp_ceiling_5",
    "fp_floor_5",
    "current_price",
]

CAT_COLUMNS = ["primary_position"]


def encode_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dummies = pd.get_dummies(out["primary_position"], prefix="pos")
    return pd.concat([out, dummies], axis=1)


def feature_columns_expanded() -> list[str]:
    base = list(FEATURE_COLUMNS)
    for p in ("DEF", "MID", "RUC", "FWD"):
        c = f"pos_{p}"
        if c not in base:
            base.append(c)
    return base
