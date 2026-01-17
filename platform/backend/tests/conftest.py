"""Pytest configuration and fixtures for backend tests."""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

import app.database as db_module
from app.config import Settings
from app.main import app

# IMPORTANT: Import all models to ensure they are registered with Base.metadata
from app.models import (  # noqa: F401
    Artifact,
    Base,
    BaseModel,
    Comparison,
    Document,
    Evaluation,
    EvaluationJob,
    EvaluationResult,
    KnowledgeBase,
    KnowledgeBaseIndex,
    KnowledgeBaseVersion,
    Project,
    RAGConfig,
    RunManifest,
    TestCase,
    TestGenerationJob,
    TestSet,
    TestTemplate,
    Webhook,
)


# Test settings override
@pytest.fixture
def test_settings() -> Settings:
    """Override settings for testing."""
    return Settings(
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        STORAGE_PATH="./test_storage",
        LOG_LEVEL="DEBUG",
        LOG_FORMAT="console",
        DEBUG=True,
    )


@pytest.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create a test database engine with shared in-memory SQLite.

    Uses a file: URI with mode=memory&cache=shared so ALL connections
    (including any that bypass dependency injection) see the same database.
    """
    # Force import of all models
    from app import models as _models  # noqa: F401

    # Use shared-cache in-memory SQLite so all connections see the same DB
    db_name = f"test_{uuid.uuid4().hex}"
    url = f"sqlite+aiosqlite:///file:{db_name}?mode=memory&cache=shared"

    engine = create_async_engine(
        url,
        connect_args={"check_same_thread": False, "uri": True},
        poolclass=StaticPool,
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def patch_app_database(test_engine: AsyncEngine) -> AsyncGenerator[None, None]:
    """Monkeypatch app.database to use the test engine.

    This ensures ANY code that directly imports from app.database
    (bypassing dependency injection) still uses the test database.
    """
    # Save originals
    original_engine = db_module.engine
    original_session_maker = db_module.async_session_maker

    # Patch with test engine
    db_module.engine = test_engine
    db_module.async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    yield

    # Restore originals
    db_module.engine = original_engine
    db_module.async_session_maker = original_session_maker


@pytest.fixture
async def db_session(
    test_engine: AsyncEngine, patch_app_database: None
) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with test_engine.connect() as connection:
        # Begin a transaction that will be rolled back
        trans = await connection.begin()

        session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            autoflush=False,
        )

        yield session

        await session.close()
        await trans.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with database session override."""
    from fastapi import FastAPI

    # Create a minimal lifespan that doesn't initialize the app's database
    @asynccontextmanager
    async def test_lifespan(test_app: FastAPI) -> AsyncGenerator[None, None]:
        yield

    # Temporarily replace the app's lifespan
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = test_lifespan

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[db_module.get_db] = override_get_db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original_lifespan
