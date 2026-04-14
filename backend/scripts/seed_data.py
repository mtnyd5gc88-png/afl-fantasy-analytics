"""Populate DB from `StatSource` (synthetic by default). Run: `cd backend && python scripts/seed_data.py`"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.etl import ETLPipeline  # noqa: E402
from app.data.sources.synthetic import SyntheticStatSource  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402


async def main() -> None:
    await init_db()
    src = SyntheticStatSource(seasons=(2023, 2024), players_per_team=10)
    etl = ETLPipeline(src)
    async with AsyncSessionLocal() as session:
        for season in (2023, 2024):
            n = await etl.ingest_season(session, season)
            print(f"season {season}: ingested {n} new player-game stat rows")
        m = await etl.ingest_injury_signals(session)
        print(f"injury signals: {m}")


if __name__ == "__main__":
    asyncio.run(main())
