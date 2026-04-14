from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FixtureRecord:
    season: int
    round_number: int
    game_date: date
    home_team_abbr: str
    away_team_abbr: str
    venue: str | None = None
    expected_total_points: float | None = None


@dataclass(frozen=True)
class PlayerGameStatRecord:
    player_external_id: str
    team_abbr: str
    opponent_abbr: str
    is_home: bool
    fantasy_points: float
    disposals: float
    kicks: float
    handballs: float
    marks: float
    tackles: float
    hitouts: float
    goals: float
    behinds: float
    clearances: float
    contested_possessions: float
    uncontested_possessions: float
    inside_50s: float
    rebound_50s: float
    clangers: float
    frees_for: float
    frees_against: float
    time_on_ground_pct: float


@dataclass(frozen=True)
class TeamGameStatRecord:
    team_abbr: str
    disposals: float
    contested_possessions: float
    inside_50s: float
    marks: float
    tackles: float
    hitouts: float
    goals: float
    possessions_share: float


@dataclass(frozen=True)
class TeamSelectionRecord:
    player_external_id: str
    change_type: str  # IN, OUT


@dataclass(frozen=True)
class InjuryNewsRecord:
    player_external_id: str | None
    player_name_guess: str | None
    status: str
    impact_score: float
    headline: str


class StatSource(ABC):
    """Pluggable data source: swap implementations for APIs, CSV, or scrapers."""

    name: str = "abstract"

    @abstractmethod
    def iter_fixtures(self, season: int) -> Iterable[FixtureRecord]:
        raise NotImplementedError

    @abstractmethod
    def player_stats_for_fixture(
        self, season: int, round_number: int, home_abbr: str, away_abbr: str
    ) -> Sequence[tuple[PlayerGameStatRecord, TeamGameStatRecord, TeamGameStatRecord]]:
        """Return (player_row, home_team_agg, away_team_agg) tuples for one match."""

    @abstractmethod
    def selections_for_fixture(
        self, season: int, round_number: int, home_abbr: str, away_abbr: str
    ) -> Sequence[TeamSelectionRecord]:
        ...

    @abstractmethod
    def injury_signals(self) -> Sequence[InjuryNewsRecord]:
        ...

    def player_roster_ext(self) -> Sequence[tuple[str, str, str]]:
        """Optional (external_id, team_abbr, position) for price/position bootstrap."""
        return ()
