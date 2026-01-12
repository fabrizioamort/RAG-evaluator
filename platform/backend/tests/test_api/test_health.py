"""Tests for health check endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_returns_200(client: AsyncClient) -> None:
    """Test that health check returns 200 when database is connected."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_check_detail(client: AsyncClient) -> None:
    """Test detailed health check endpoint."""
    response = await client.get("/api/v1/health/detail")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "database_type" in data
    assert "storage_path" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """Test root endpoint returns welcome message."""
    response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "health" in data


@pytest.mark.asyncio
async def test_request_id_header(client: AsyncClient) -> None:
    """Test that request ID header is returned."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
