"""Incremental ETL stub. Run: `cd backend && python scripts/daily_update.py`"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.jobs import daily_incremental_update  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402


async def main() -> None:
    await init_db()
    async with AsyncSessionLocal() as session:
        r = await daily_incremental_update(session)
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
