"""Pytest configuration and fixtures for backend tests."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Generator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.database import Base, get_db
from app.main import app


# Test settings override
@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for testing."""
    return Settings(
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        STORAGE_PATH="./test_storage",
        LOG_LEVEL="DEBUG",
        LOG_FORMAT="console",
        DEBUG=True,
    )





@pytest.fixture(scope="session")
async def test_engine(test_settings: Settings) -> AsyncGenerator[AsyncEngine, None]:
    """Create a test database engine."""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_connection(test_engine: AsyncEngine) -> AsyncGenerator[AsyncConnection, None]:
    """Create a test database connection with transaction for isolation."""
    async with test_engine.connect() as connection:
        # Start outer transaction that will be rolled back
        transaction = await connection.begin()
        yield connection
        # Rollback the outer transaction to undo all changes
        await transaction.rollback()


@pytest.fixture
async def db_session(db_connection: AsyncConnection) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session using nested transactions for isolation.

    Uses savepoints so that when the API calls commit(), it only commits
    to the savepoint. The outer transaction is rolled back at the end.
    """
    # Start a nested transaction (savepoint)
    nested = await db_connection.begin_nested()

    session = AsyncSession(
        bind=db_connection,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    yield session

    await session.close()
    # Rollback the nested transaction
    await nested.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with database session override."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
