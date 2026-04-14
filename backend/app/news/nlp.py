"""
Lightweight rule + pattern NLP for availability extraction.

Production: swap in NER / classifier; this layer stays as a deterministic baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Player, PlayerAvailabilitySignal


@dataclass(frozen=True)
class ParsedAvailability:
    player_name: str | None
    status: str  # IN, OUT, QUESTIONABLE
    impact_score: float
    headline: str


_STATUS_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"\b(ruled out|won't play|will miss|omitted|omitted from|late out)\b", re.I), "OUT", 0.95),
    (re.compile(r"\b(out indefinitely|season ending|surgery)\b", re.I), "OUT", 1.0),
    (re.compile(r"\b(test|late change|game time decision|in doubt|questionable)\b", re.I), "QUESTIONABLE", 0.5),
    (re.compile(r"\b(named in|returns?|cleared to play|back in the side|will play)\b", re.I), "IN", 0.2),
]


def extract_availability_from_text(text: str) -> ParsedAvailability | None:
    t = text.strip()
    if not t:
        return None
    status = "QUESTIONABLE"
    impact = 0.4
    for pat, st, imp in _STATUS_PATTERNS:
        if pat.search(t):
            status = st
            impact = imp
            break
    name = None
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", t)
    if m:
        name = m.group(1)
    return ParsedAvailability(player_name=name, status=status, impact_score=impact, headline=t[:500])


async def ingest_text_signals(session: AsyncSession, texts: list[str], source: str = "nlp") -> int:
    n = 0
    for tx in texts:
        p = extract_availability_from_text(tx)
        if not p or not p.player_name:
            continue
        r = await session.execute(select(Player).where(Player.name.ilike(f"%{p.player_name}%")))
        pl = r.scalar_one_or_none()
        if not pl:
            continue
        session.add(
            PlayerAvailabilitySignal(
                player_id=pl.id,
                source=source,
                status=p.status,
                impact_score=p.impact_score,
                headline=p.headline,
            )
        )
        n += 1
    await session.commit()
    return n
