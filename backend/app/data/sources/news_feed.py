"""Optional HTML feed adapter implementing `StatSource.injury_signals` via `httpx` + NLP."""

from __future__ import annotations

from typing import Iterable, Sequence

import httpx
from bs4 import BeautifulSoup

from app.data.sources.base import (
    FixtureRecord,
    InjuryNewsRecord,
    PlayerGameStatRecord,
    StatSource,
    TeamGameStatRecord,
    TeamSelectionRecord,
)
from app.news.nlp import extract_availability_from_text


class NewsOnlyFeedSource(StatSource):
    """Fetches public injury headlines; pair with a stats source for full ETL."""

    name = "news_feed"

    def __init__(self, urls: Sequence[str]) -> None:
        self.urls = list(urls)

    def iter_fixtures(self, season: int) -> Iterable[FixtureRecord]:
        return []

    def player_stats_for_fixture(
        self, season: int, round_number: int, home_abbr: str, away_abbr: str
    ) -> Sequence[tuple[PlayerGameStatRecord, TeamGameStatRecord, TeamGameStatRecord]]:
        return []

    def selections_for_fixture(
        self, season: int, round_number: int, home_abbr: str, away_abbr: str
    ) -> Sequence[TeamSelectionRecord]:
        return []

    def injury_signals(self) -> Sequence[InjuryNewsRecord]:
        out: list[InjuryNewsRecord] = []
        for url in self.urls:
            try:
                r = httpx.get(url, timeout=15.0, follow_redirects=True)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                text = " ".join(soup.stripped_strings)[:50_000]
            except Exception:
                continue
            for chunk in text.split(".")[:200]:
                p = extract_availability_from_text(chunk + ".")
                if not p:
                    continue
                out.append(
                    InjuryNewsRecord(
                        player_external_id=None,
                        player_name_guess=p.player_name,
                        status=p.status,
                        impact_score=p.impact_score,
                        headline=p.headline,
                    )
                )
        return out
