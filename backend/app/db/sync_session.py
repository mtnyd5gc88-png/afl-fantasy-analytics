"""Synchronous engine for batch training / pandas (avoids async sqlite limits)."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def _to_sync_url(url: str) -> str:
    if "+aiosqlite" in url:
        return url.replace("sqlite+aiosqlite", "sqlite", 1)
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg2")
    return url


sync_engine = create_engine(_to_sync_url(settings.database_url), future=True)
SyncSessionLocal = sessionmaker(bind=sync_engine, class_=Session, autoflush=False, expire_on_commit=False)


def get_sync_session() -> Session:
    return SyncSessionLocal()
