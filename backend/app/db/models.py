from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(8), unique=True, nullable=False)


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    primary_position: Mapped[str] = mapped_column(String(8), nullable=False)  # DEF, MID, RUC, FWD
    current_price: Mapped[float] = mapped_column(Float, default=300_000.0)

    team: Mapped[Team] = relationship(backref="players")


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    venue: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    expected_total_points: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (UniqueConstraint("season", "round_number", "home_team_id", "away_team_id"),)


class PlayerGameStat(Base):
    """Normalized per-game player statistics (AFL Fantasy relevant)."""

    __tablename__ = "player_game_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    opponent_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, default=False)
    fantasy_points: Mapped[float] = mapped_column(Float, nullable=False)

    disposals: Mapped[float] = mapped_column(Float, default=0)
    kicks: Mapped[float] = mapped_column(Float, default=0)
    handballs: Mapped[float] = mapped_column(Float, default=0)
    marks: Mapped[float] = mapped_column(Float, default=0)
    tackles: Mapped[float] = mapped_column(Float, default=0)
    hitouts: Mapped[float] = mapped_column(Float, default=0)
    goals: Mapped[float] = mapped_column(Float, default=0)
    behinds: Mapped[float] = mapped_column(Float, default=0)
    clearances: Mapped[float] = mapped_column(Float, default=0)
    contested_possessions: Mapped[float] = mapped_column(Float, default=0)
    uncontested_possessions: Mapped[float] = mapped_column(Float, default=0)
    inside_50s: Mapped[float] = mapped_column(Float, default=0)
    rebound_50s: Mapped[float] = mapped_column(Float, default=0)
    clangers: Mapped[float] = mapped_column(Float, default=0)
    frees_for: Mapped[float] = mapped_column(Float, default=0)
    frees_against: Mapped[float] = mapped_column(Float, default=0)
    time_on_ground_pct: Mapped[float] = mapped_column(Float, default=80.0)

    __table_args__ = (UniqueConstraint("player_id", "fixture_id"),)


class TeamGameStat(Base):
    __tablename__ = "team_game_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), nullable=False)
    disposals: Mapped[float] = mapped_column(Float, default=0)
    contested_possessions: Mapped[float] = mapped_column(Float, default=0)
    inside_50s: Mapped[float] = mapped_column(Float, default=0)
    marks: Mapped[float] = mapped_column(Float, default=0)
    tackles: Mapped[float] = mapped_column(Float, default=0)
    hitouts: Mapped[float] = mapped_column(Float, default=0)
    goals: Mapped[float] = mapped_column(Float, default=0)
    possessions_share: Mapped[float] = mapped_column(Float, default=0.5)

    __table_args__ = (UniqueConstraint("team_id", "fixture_id"),)


class TeamSelection(Base):
    __tablename__ = "team_selections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), nullable=False)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    change_type: Mapped[str] = mapped_column(String(8), nullable=False)  # IN, OUT

    __table_args__ = (UniqueConstraint("fixture_id", "player_id", "change_type"),)


class PlayerAvailabilitySignal(Base):
    __tablename__ = "player_availability_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    source: Mapped[str] = mapped_column(String(64), default="nlp")
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # IN, OUT, QUESTIONABLE
    impact_score: Mapped[float] = mapped_column(Float, default=0.0)
    headline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metric_rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    path_json: Mapped[str] = mapped_column(String(512), nullable=False)
