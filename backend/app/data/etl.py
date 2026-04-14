from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.sources.base import StatSource
from app.db.models import (
    Fixture,
    Player,
    PlayerAvailabilitySignal,
    PlayerGameStat,
    Team,
    TeamGameStat,
    TeamSelection,
)


class ETLPipeline:
    """Loads normalized stats from a `StatSource` into the database."""

    def __init__(self, source: StatSource) -> None:
        self.source = source

    async def _ensure_team(self, session: AsyncSession, abbr: str, name: str | None = None) -> Team:
        r = await session.execute(select(Team).where(Team.abbreviation == abbr))
        row = r.scalar_one_or_none()
        if row:
            return row
        t = Team(name=name or abbr, abbreviation=abbr)
        session.add(t)
        await session.flush()
        return t

    async def _ensure_player(
        self,
        session: AsyncSession,
        external_id: str,
        team: Team,
        position: str,
        price: float,
    ) -> Player:
        # external_id stored as name suffix pattern: use name = external_id for synthetic
        r = await session.execute(select(Player).where(Player.name == external_id))
        p = r.scalar_one_or_none()
        if p:
            p.team_id = team.id
            p.primary_position = position
            p.current_price = price
            return p
        p = Player(team_id=team.id, name=external_id, primary_position=position, current_price=price)
        session.add(p)
        await session.flush()
        return p

    async def ingest_season(self, session: AsyncSession, season: int) -> int:
        count = 0
        player_pos_map: dict[str, str] = {ext: pos for ext, _, pos in self.source.player_roster_ext()}

        for fx in self.source.iter_fixtures(season):
            home = await self._ensure_team(session, fx.home_team_abbr)
            away = await self._ensure_team(session, fx.away_team_abbr)
            r = await session.execute(
                select(Fixture).where(
                    Fixture.season == fx.season,
                    Fixture.round_number == fx.round_number,
                    Fixture.home_team_id == home.id,
                    Fixture.away_team_id == away.id,
                )
            )
            fixture = r.scalar_one_or_none()
            if fixture is None:
                fixture = Fixture(
                    season=fx.season,
                    round_number=fx.round_number,
                    game_date=fx.game_date,
                    home_team_id=home.id,
                    away_team_id=away.id,
                    venue=fx.venue,
                    expected_total_points=fx.expected_total_points,
                )
                session.add(fixture)
                await session.flush()

            triples = self.source.player_stats_for_fixture(
                season, fx.round_number, fx.home_team_abbr, fx.away_team_abbr
            )
            for pstat, home_agg, away_agg in triples:
                team_abbr = pstat.team_abbr
                team = home if team_abbr == fx.home_team_abbr else away
                opp = away if team_abbr == fx.home_team_abbr else home
                pos = player_pos_map.get(pstat.player_external_id, "MID")
                price = 300_000.0 + hash(pstat.player_external_id) % 250_000
                player = await self._ensure_player(session, pstat.player_external_id, team, pos, float(price))

                q = await session.execute(
                    select(PlayerGameStat).where(
                        PlayerGameStat.player_id == player.id,
                        PlayerGameStat.fixture_id == fixture.id,
                    )
                )
                if q.scalar_one_or_none():
                    continue

                pgs = PlayerGameStat(
                    player_id=player.id,
                    fixture_id=fixture.id,
                    team_id=team.id,
                    opponent_team_id=opp.id,
                    is_home=pstat.is_home,
                    fantasy_points=pstat.fantasy_points,
                    disposals=pstat.disposals,
                    kicks=pstat.kicks,
                    handballs=pstat.handballs,
                    marks=pstat.marks,
                    tackles=pstat.tackles,
                    hitouts=pstat.hitouts,
                    goals=pstat.goals,
                    behinds=pstat.behinds,
                    clearances=pstat.clearances,
                    contested_possessions=pstat.contested_possessions,
                    uncontested_possessions=pstat.uncontested_possessions,
                    inside_50s=pstat.inside_50s,
                    rebound_50s=pstat.rebound_50s,
                    clangers=pstat.clangers,
                    frees_for=pstat.frees_for,
                    frees_against=pstat.frees_against,
                    time_on_ground_pct=pstat.time_on_ground_pct,
                )
                session.add(pgs)
                count += 1

            for agg, tid in ((home_agg, home.id), (away_agg, away.id)):
                q = await session.execute(
                    select(TeamGameStat).where(
                        TeamGameStat.team_id == tid,
                        TeamGameStat.fixture_id == fixture.id,
                    )
                )
                if q.scalar_one_or_none():
                    continue
                session.add(
                    TeamGameStat(
                        team_id=tid,
                        fixture_id=fixture.id,
                        disposals=agg.disposals,
                        contested_possessions=agg.contested_possessions,
                        inside_50s=agg.inside_50s,
                        marks=agg.marks,
                        tackles=agg.tackles,
                        hitouts=agg.hitouts,
                        goals=agg.goals,
                        possessions_share=agg.possessions_share,
                    )
                )

            for sel in self.source.selections_for_fixture(
                season, fx.round_number, fx.home_team_abbr, fx.away_team_abbr
            ):
                pl = await session.execute(select(Player).where(Player.name == sel.player_external_id))
                pl_row = pl.scalar_one_or_none()
                if not pl_row:
                    continue
                q = await session.execute(
                    select(TeamSelection).where(
                        TeamSelection.fixture_id == fixture.id,
                        TeamSelection.player_id == pl_row.id,
                        TeamSelection.change_type == sel.change_type,
                    )
                )
                if q.scalar_one_or_none():
                    continue
                session.add(
                    TeamSelection(
                        fixture_id=fixture.id,
                        player_id=pl_row.id,
                        change_type=sel.change_type,
                    )
                )

        await session.commit()
        return count

    async def ingest_injury_signals(self, session: AsyncSession) -> int:
        n = 0
        for sig in self.source.injury_signals():
            pid = None
            if sig.player_external_id:
                r = await session.execute(select(Player).where(Player.name == sig.player_external_id))
                p = r.scalar_one_or_none()
                if p:
                    pid = p.id
            if pid is None and sig.player_name_guess:
                r = await session.execute(select(Player).where(Player.name.ilike(f"%{sig.player_name_guess}%")))
                p = r.scalar_one_or_none()
                if p:
                    pid = p.id
            if pid is None:
                continue
            session.add(
                PlayerAvailabilitySignal(
                    player_id=pid,
                    source=self.source.name,
                    status=sig.status,
                    impact_score=sig.impact_score,
                    headline=sig.headline,
                )
            )
            n += 1
        await session.commit()
        return n
