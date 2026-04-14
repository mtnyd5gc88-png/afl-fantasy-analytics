from __future__ import annotations

import math

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    HealthOut,
    PlayerProjectionOut,
    PredictRequest,
    TeamEvaluateRequest,
    TeamEvaluateResponse,
    TradeRecommendOut,
    TradeRecommendRequest,
)
from app.db.session import get_async_session
from app.ml.predict import PredictionService, PlayerProjection
from app.optimization.trade_optimizer import TradeOptimizer

router = APIRouter()

_svc: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _svc
    if _svc is None:
        _svc = PredictionService()
    return _svc


def _to_out(p: PlayerProjection) -> PlayerProjectionOut:
    return PlayerProjectionOut(
        player_id=p.player_id,
        name=p.name,
        team_id=p.team_id,
        position=p.position,
        price=p.price,
        predicted_points=p.predicted_points,
        ci_low=p.ci_low,
        ci_high=p.ci_high,
        variance_proxy=p.variance_proxy,
        value_vs_price=p.value_vs_price,
    )


@router.get("/health", response_model=HealthOut)
async def health(svc: PredictionService = Depends(get_prediction_service)) -> HealthOut:
    return HealthOut(ok=True, model_ready=svc.is_ready())


@router.get("/players", response_model=list[PlayerProjectionOut])
async def list_players(svc: PredictionService = Depends(get_prediction_service)) -> list[PlayerProjectionOut]:
    if not svc.is_ready():
        raise HTTPException(503, "Model not trained. Run scripts/train_model.py")
    projs = svc.project_all()
    return [_to_out(p) for p in projs]


@router.post("/predict", response_model=list[PlayerProjectionOut])
async def predict(req: PredictRequest, svc: PredictionService = Depends(get_prediction_service)) -> list[PlayerProjectionOut]:
    if not svc.is_ready():
        raise HTTPException(503, "Model not trained")
    latest = svc.build_latest_feature_rows()
    sub = latest[latest["player_id"].isin(req.player_ids)]
    if sub.empty:
        raise HTTPException(404, "No matching players in feature store")
    pred, _ = svc.predict_frame(sub)
    n_std = 5
    out: list[PlayerProjectionOut] = []
    for i, row in sub.reset_index(drop=True).iterrows():
        p = float(pred[i])
        fp_std = float(row.get(f"fp_std_{n_std}", 0) or 0)
        lo, hi = svc.interval(p, fp_std)
        price = float(row.get("current_price", 300_000))
        out.append(
            PlayerProjectionOut(
                player_id=int(row["player_id"]),
                name=str(row["player_name"]),
                team_id=int(row["player_team_id"]),
                position=str(row["primary_position"]),
                price=price,
                predicted_points=p,
                ci_low=lo,
                ci_high=hi,
                variance_proxy=fp_std,
                value_vs_price=p / max(price / 100_000.0, 1e-6),
            )
        )
    return out


@router.post("/team-evaluate", response_model=TeamEvaluateResponse)
async def team_evaluate(
    req: TeamEvaluateRequest, svc: PredictionService = Depends(get_prediction_service)
) -> TeamEvaluateResponse:
    if not svc.is_ready():
        raise HTTPException(503, "Model not trained")
    if len(req.player_ids) != len(set(req.player_ids)):
        raise HTTPException(400, "Duplicate player_ids")
    latest = svc.build_latest_feature_rows()
    sub = latest[latest["player_id"].isin(req.player_ids)]
    if len(sub) != len(set(req.player_ids)):
        raise HTTPException(400, "Some player_ids missing from dataset")
    pred, _ = svc.predict_frame(sub)
    n_std = 5
    per: list[PlayerProjectionOut] = []
    means: list[float] = []
    sigs: list[float] = []
    for i, row in sub.reset_index(drop=True).iterrows():
        p = float(pred[i])
        fp_std = float(row.get(f"fp_std_{n_std}", 0) or 0)
        lo, hi = svc.interval(p, fp_std)
        price = float(row.get("current_price", 300_000))
        per.append(
            PlayerProjectionOut(
                player_id=int(row["player_id"]),
                name=str(row["player_name"]),
                team_id=int(row["player_team_id"]),
                position=str(row["primary_position"]),
                price=price,
                predicted_points=p,
                ci_low=lo,
                ci_high=hi,
                variance_proxy=fp_std,
                value_vs_price=p / max(price / 100_000.0, 1e-6),
            )
        )
        means.append(p)
        sigs.append(math.sqrt(svc.sigma_global**2 + (0.35 * fp_std) ** 2))
    exp = float(sum(means))
    team_sig = float(math.sqrt(sum(s * s for s in sigs)))
    z = 1.645
    return TeamEvaluateResponse(
        expected_total=exp,
        ci_low=exp - z * team_sig,
        ci_high=exp + z * team_sig,
        per_player=per,
    )


@router.post("/trade-recommend", response_model=list[TradeRecommendOut])
async def trade_recommend(
    req: TradeRecommendRequest, svc: PredictionService = Depends(get_prediction_service)
) -> list[TradeRecommendOut]:
    if not svc.is_ready():
        raise HTTPException(503, "Model not trained")
    opt = TradeOptimizer(svc)
    try:
        recs = opt.recommend(
            field_player_ids=req.field_player_ids,
            bank=req.bank,
            max_trades=req.max_trades,
            horizon_rounds=req.horizon_rounds,
            mc_samples=req.mc_samples,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    return [
        TradeRecommendOut(
            trade_out_id=r.trade_out_id,
            trade_in_id=r.trade_in_id,
            expected_value_gain=r.expected_value_gain,
            risk_team_delta_std=r.risk_team_delta_std,
            net_budget_delta=r.net_budget_delta,
            mc_mean=r.mc_mean,
            mc_std=r.mc_std,
        )
        for r in recs
    ]


@router.post("/ingest-news")
async def ingest_news(
    texts: list[str] = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, int]:
    from app.news.nlp import ingest_text_signals

    n = await ingest_text_signals(session, texts)
    return {"signals_written": n}
