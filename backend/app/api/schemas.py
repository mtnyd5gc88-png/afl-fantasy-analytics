from __future__ import annotations

from pydantic import BaseModel, Field


class PlayerProjectionOut(BaseModel):
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


class PredictRequest(BaseModel):
    player_ids: list[int] = Field(..., min_length=1)


class TeamEvaluateRequest(BaseModel):
    player_ids: list[int] = Field(..., min_length=1)


class TeamEvaluateResponse(BaseModel):
    expected_total: float
    ci_low: float
    ci_high: float
    per_player: list[PlayerProjectionOut]


class TradeRecommendRequest(BaseModel):
    field_player_ids: list[int] = Field(..., min_length=3)
    bank: float = Field(..., ge=0)
    max_trades: int = Field(1, ge=1, le=5)
    horizon_rounds: int = Field(3, ge=1, le=15)
    mc_samples: int = Field(2000, ge=200, le=20_000)


class TradeRecommendOut(BaseModel):
    trade_out_id: int
    trade_in_id: int
    expected_value_gain: float
    risk_team_delta_std: float
    net_budget_delta: float
    mc_mean: float
    mc_std: float


class HealthOut(BaseModel):
    ok: bool
    model_ready: bool
