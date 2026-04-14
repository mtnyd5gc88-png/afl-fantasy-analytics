"""Scheduled-style jobs (invoke via `python -m app.scripts.daily_update` or APScheduler in prod)."""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.data.etl import ETLPipeline
from app.data.sources.synthetic import SyntheticStatSource

log = logging.getLogger(__name__)


async def daily_incremental_update(session: AsyncSession, season: int | None = None) -> dict[str, int]:
    """
    Re-run ingestion for current season. Idempotent on (player, fixture) keys.
    For production, plug a live `StatSource` that fetches deltas since `last_run`.
    """
    src = SyntheticStatSource()
    etl = ETLPipeline(src)
    seasons = (season,) if season else (2023, 2024)
    total = 0
    for yr in seasons:
        total += await etl.ingest_season(session, yr)
    injuries = await etl.ingest_injury_signals(session)
    log.info("daily_update seasons=%s player_game_rows=%s injury_signals=%s", seasons, total, injuries)
    return {"player_game_rows_upserted": total, "injury_signals": injuries}
