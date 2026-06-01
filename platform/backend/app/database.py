"""Database configuration with SQLAlchemy async engine."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, StaticPool  # StaticPool kept for :memory: test DBs

from app.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""

    pass


def get_engine_options() -> dict[str, Any]:
    """Get database engine options based on database type."""
    if settings.is_sqlite:
        pool_class = StaticPool if ":memory:" in settings.DATABASE_URL else NullPool
        return {
            "connect_args": {"check_same_thread": False},
            "poolclass": pool_class,
            "echo": settings.DEBUG,
        }
    else:
        # PostgreSQL options
        return {
            "poolclass": NullPool,  # Use NullPool for async
            "echo": settings.DEBUG,
        }


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    **get_engine_options(),
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides a database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions (for non-request contexts)."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_db_connection() -> bool:
    """Check if database connection is healthy."""
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception:
        return False
