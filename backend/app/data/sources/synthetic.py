"""
Synthetic AFL-style data generator for end-to-end development.

**Explicitly marked:** Rows are simulated from plausible distributions, not scraped AFL.
Interfaces match real ingestion; swap `StatSource` implementation for production feeds.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from typing import Iterable, Sequence

import numpy as np

from app.data.sources.base import (
    FixtureRecord,
    InjuryNewsRecord,
    PlayerGameStatRecord,
    StatSource,
    TeamGameStatRecord,
    TeamSelectionRecord,
)


def _stable_seed(*parts: str) -> int:
    h = hashlib.sha256("".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def _role_base_stats(pos: str, rng: np.random.Generator) -> dict[str, float]:
    if pos == "RUC":
        return {
            "kicks": rng.normal(8, 3),
            "handballs": rng.normal(6, 2),
            "marks": rng.normal(3, 1.5),
            "tackles": rng.normal(4, 2),
            "hitouts": rng.normal(28, 8),
            "goals": max(0, rng.normal(0.4, 0.5)),
            "behinds": max(0, rng.normal(0.2, 0.3)),
            "clearances": rng.normal(6, 2),
            "contested_possessions": rng.normal(10, 3),
            "uncontested_possessions": rng.normal(6, 2),
            "inside_50s": rng.normal(2, 1),
            "rebound_50s": rng.normal(1, 0.8),
            "clangers": rng.normal(2, 1),
            "frees_for": rng.normal(1, 0.8),
            "frees_against": rng.normal(1.5, 0.8),
            "tog": float(np.clip(rng.normal(88, 6), 40, 98)),
        }
    if pos == "DEF":
        return {
            "kicks": rng.normal(14, 4),
            "handballs": rng.normal(8, 3),
            "marks": rng.normal(6, 2),
            "tackles": rng.normal(5, 2),
            "hitouts": rng.normal(0.2, 0.4),
            "goals": max(0, rng.normal(0.15, 0.3)),
            "behinds": max(0, rng.normal(0.2, 0.3)),
            "clearances": rng.normal(2, 1),
            "contested_possessions": rng.normal(6, 2),
            "uncontested_possessions": rng.normal(10, 3),
            "inside_50s": rng.normal(3, 1.2),
            "rebound_50s": rng.normal(5, 2),
            "clangers": rng.normal(2.5, 1),
            "frees_for": rng.normal(1, 0.7),
            "frees_against": rng.normal(1.2, 0.7),
            "tog": float(np.clip(rng.normal(86, 7), 40, 98)),
        }
    if pos == "FWD":
        return {
            "kicks": rng.normal(10, 3),
            "handballs": rng.normal(7, 2.5),
            "marks": rng.normal(5, 2),
            "tackles": rng.normal(4, 2),
            "hitouts": rng.normal(0.1, 0.2),
            "goals": max(0, rng.normal(1.2, 0.9)),
            "behinds": max(0, rng.normal(0.8, 0.5)),
            "clearances": rng.normal(2, 1),
            "contested_possessions": rng.normal(7, 2.5),
            "uncontested_possessions": rng.normal(8, 2.5),
            "inside_50s": rng.normal(4, 1.5),
            "rebound_50s": rng.normal(1, 0.8),
            "clangers": rng.normal(2, 1),
            "frees_for": rng.normal(1.2, 0.8),
            "frees_against": rng.normal(1.3, 0.8),
            "tog": float(np.clip(rng.normal(82, 8), 40, 98)),
        }
    # MID
    return {
        "kicks": rng.normal(12, 3.5),
        "handballs": rng.normal(12, 3.5),
        "marks": rng.normal(5, 2),
        "tackles": rng.normal(6, 2.5),
        "hitouts": rng.normal(0.5, 0.8),
        "goals": max(0, rng.normal(0.5, 0.5)),
        "behinds": max(0, rng.normal(0.4, 0.4)),
        "clearances": rng.normal(5, 2),
        "contested_possessions": rng.normal(9, 3),
        "uncontested_possessions": rng.normal(11, 3),
        "inside_50s": rng.normal(4, 1.5),
        "rebound_50s": rng.normal(2, 1),
        "clangers": rng.normal(2.5, 1),
        "frees_for": rng.normal(1.1, 0.8),
        "frees_against": rng.normal(1.3, 0.8),
        "tog": float(np.clip(rng.normal(84, 7), 40, 98)),
    }


class SyntheticStatSource(StatSource):
    """Generates multi-season schedules and correlated player/team stats."""

    name = "synthetic"

    def __init__(
        self,
        seasons: tuple[int, ...] = (2023, 2024),
        teams: Sequence[str] | None = None,
        players_per_team: int = 28,
        seed: int = 42,
    ) -> None:
        self.seasons = seasons
        self.teams = list(teams or self._default_teams())
        self.players_per_team = players_per_team
        self.seed = seed
        self._player_meta: list[tuple[str, str, str]] = []  # (ext_id, team, pos)
        self._build_roster()

    @staticmethod
    def _default_teams() -> list[str]:
        return [
            "ADE",
            "BRI",
            "CAR",
            "COL",
            "ESS",
            "FRE",
            "GEE",
            "GCS",
            "GWS",
            "HAW",
            "MEL",
            "NTH",
            "PTA",
            "RIC",
            "STK",
            "SYD",
            "WBD",
            "WCE",
        ]

    def _build_roster(self) -> None:
        positions = ["DEF", "DEF", "MID", "MID", "MID", "RUC", "FWD", "FWD"]
        idx = 0
        for t in self.teams:
            for _ in range(self.players_per_team):
                pos = positions[idx % len(positions)]
                idx += 1
                ext = f"{t}-{pos}-{idx}"
                self._player_meta.append((ext, t, pos))

    def iter_fixtures(self, season: int) -> Iterable[FixtureRecord]:
        rng = np.random.default_rng(_stable_seed("fixtures", str(season)))
        start = date(season, 3, 15)
        for rnd in range(1, 24):
            pairs = list(zip(self.teams[::2], self.teams[1::2]))
            rng.shuffle(pairs)
            for hi, ai in pairs:
                day_offset = int(rng.integers(0, 4))
                gd = start + timedelta(days=(rnd - 1) * 7 + day_offset)
                total = float(rng.normal(165, 12))
                yield FixtureRecord(
                    season=season,
                    round_number=rnd,
                    game_date=gd,
                    home_team_abbr=hi,
                    away_team_abbr=ai,
                    venue=None,
                    expected_total_points=max(120.0, total),
                )

    def player_stats_for_fixture(
        self, season: int, round_number: int, home_abbr: str, away_abbr: str
    ) -> Sequence[tuple[PlayerGameStatRecord, TeamGameStatRecord, TeamGameStatRecord]]:
        rng = np.random.default_rng(
            _stable_seed("pgs", str(season), str(round_number), home_abbr, away_abbr)
        )

        def sim_side(team_abbr: str, opp: str, is_home: bool) -> list[PlayerGameStatRecord]:
            rows: list[PlayerGameStatRecord] = []
            team_players = [(e, t, p) for e, t, p in self._player_meta if t == team_abbr]
            opp_strength = 0.02 * (hash(opp) % 17 - 8)
            pace_boost = 0.01 * (rng.normal(0, 1))

            for ext_id, _, pos in team_players:
                prng = np.random.default_rng(_stable_seed(ext_id, str(season), str(round_number)))
                base = _role_base_stats(pos, prng)
                # opponent / pace modifiers
                scale = 1.0 + pace_boost - opp_strength * (1.0 if pos in ("MID", "FWD") else 0.6)
                if is_home:
                    scale += 0.02

                kicks = max(0, base["kicks"] * scale + prng.normal(0, 2))
                handballs = max(0, base["handballs"] * scale + prng.normal(0, 2))
                marks = max(0, base["marks"] * scale + prng.normal(0, 1.5))
                tackles = max(0, base["tackles"] + prng.normal(0, 2))
                hitouts = max(0, base["hitouts"] * (1.1 if pos == "RUC" else 1.0) + prng.normal(0, 3))
                goals = max(0, base["goals"] + prng.normal(0, 0.4))
                behinds = max(0, base["behinds"] + prng.normal(0, 0.3))
                clearances = max(0, base["clearances"] + prng.normal(0, 1.5))
                cp = max(0, base["contested_possessions"] * scale + prng.normal(0, 2))
                ucp = max(0, base["uncontested_possessions"] * scale + prng.normal(0, 2))
                i50 = max(0, base["inside_50s"] + prng.normal(0, 1.2))
                r50 = max(0, base["rebound_50s"] + prng.normal(0, 1))
                clangers = max(0, base["clangers"] + prng.normal(0, 1))
                ff = max(0, base["frees_for"] + prng.normal(0, 0.8))
                fa = max(0, base["frees_against"] + prng.normal(0, 0.8))
                tog = float(np.clip(base["tog"] + prng.normal(0, 5), 20, 100))

                fp = (
                    (kicks + handballs) * 3
                    + marks * 3
                    + tackles * 4
                    + hitouts * 1.5
                    + goals * 6
                    + behinds
                    + ff
                    - fa * 3
                    - clangers * 3
                    + cp * 0.5
                )
                fp = max(0, fp + prng.normal(0, 8))

                rows.append(
                    PlayerGameStatRecord(
                        player_external_id=ext_id,
                        team_abbr=team_abbr,
                        opponent_abbr=opp,
                        is_home=is_home,
                        fantasy_points=float(fp),
                        disposals=float(kicks + handballs),
                        kicks=float(kicks),
                        handballs=float(handballs),
                        marks=float(marks),
                        tackles=float(tackles),
                        hitouts=float(hitouts),
                        goals=float(goals),
                        behinds=float(behinds),
                        clearances=float(clearances),
                        contested_possessions=float(cp),
                        uncontested_possessions=float(ucp),
                        inside_50s=float(i50),
                        rebound_50s=float(r50),
                        clangers=float(clangers),
                        frees_for=float(ff),
                        frees_against=float(fa),
                        time_on_ground_pct=float(tog),
                    )
                )
            return rows

        home_rows = sim_side(home_abbr, away_abbr, True)
        away_rows = sim_side(away_abbr, home_abbr, False)

        def agg(rows: list[PlayerGameStatRecord], opp_disposals: float) -> TeamGameStatRecord:
            disposals = sum(r.kicks + r.handballs for r in rows)
            cp = sum(r.contested_possessions for r in rows)
            i50 = sum(r.inside_50s for r in rows)
            marks = sum(r.marks for r in rows)
            tackles = sum(r.tackles for r in rows)
            hitouts = sum(r.hitouts for r in rows)
            goals = sum(r.goals for r in rows)
            denom = disposals + opp_disposals + 1e-6
            share = float(np.clip(disposals / denom, 0.35, 0.65))
            return TeamGameStatRecord(
                team_abbr=rows[0].team_abbr if rows else "",
                disposals=float(disposals),
                contested_possessions=float(cp),
                inside_50s=float(i50),
                marks=float(marks),
                tackles=float(tackles),
                hitouts=float(hitouts),
                goals=float(goals),
                possessions_share=share,
            )

        home_disp = sum(r.kicks + r.handballs for r in home_rows)
        away_disp = sum(r.kicks + r.handballs for r in away_rows)
        home_agg = agg(home_rows, away_disp)
        away_agg = agg(away_rows, home_disp)
        out: list[tuple[PlayerGameStatRecord, TeamGameStatRecord, TeamGameStatRecord]] = []
        for r in home_rows:
            out.append((r, home_agg, away_agg))
        for r in away_rows:
            out.append((r, away_agg, home_agg))
        return out

    def selections_for_fixture(
        self, season: int, round_number: int, home_abbr: str, away_abbr: str
    ) -> Sequence[TeamSelectionRecord]:
        rng = random.Random(_stable_seed("sel", str(season), str(round_number), home_abbr, away_abbr))
        team_players = [e for e, t, _ in self._player_meta if t in (home_abbr, away_abbr)]
        rng.shuffle(team_players)
        picks = team_players[:4]
        return [TeamSelectionRecord(p, "IN" if i % 2 == 0 else "OUT") for i, p in enumerate(picks)]

    def player_roster_ext(self) -> Sequence[tuple[str, str, str]]:
        return self._player_meta

    def injury_signals(self) -> Sequence[InjuryNewsRecord]:
        rng = random.Random(self.seed)
        signals: list[InjuryNewsRecord] = []
        for ext, _, _ in self._player_meta[::40]:
            roll = rng.random()
            if roll < 0.15:
                status = "OUT"
                impact = 0.9
            elif roll < 0.35:
                status = "QUESTIONABLE"
                impact = 0.45
            else:
                continue
            signals.append(
                InjuryNewsRecord(
                    player_external_id=ext,
                    player_name_guess=None,
                    status=status,
                    impact_score=impact,
                    headline=f"Synthetic: {ext} listed as {status}",
                )
            )
        return signals
