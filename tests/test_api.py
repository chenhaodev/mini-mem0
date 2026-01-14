"""Tests for API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from main import app


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Homecare Memory API"
        assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_add_memory_endpoint_validation():
    """Test add memory endpoint input validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Invalid request: empty conversation
        response = await client.post(
            "/api/v1/memories",
            json={
                "patient_id": "patient_123",
                "conversation": []
            }
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_search_endpoint_validation():
    """Test search endpoint input validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Invalid request: missing required fields
        response = await client.post(
            "/api/v1/memories/search",
            json={"patient_id": "patient_123"}  # Missing query
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_update_memory_endpoint_validation():
    """Test update memory endpoint validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Invalid memory ID format
        response = await client.patch(
            "/api/v1/memories/invalid-uuid",
            json={"content": "Updated content"}
        )

        assert response.status_code == 422  # Validation error


# Note: Full integration tests with database would require:
# 1. Test database setup
# 2. Mock or real OpenAI API
# 3. Vector store initialization
# These are covered in test_integration.py instead
