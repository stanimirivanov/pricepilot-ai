"""Tests for review API using pytest-asyncio"""

import pytest
from httpx import ASGITransport, AsyncClient

from pricepilot.api.main import app


@pytest.fixture
async def client():
    """Create async test client"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_root(client):
    """Test root endpoint"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_health(client):
    """Test health endpoint"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_get_recommendation(client):
    """Test getting recommendation"""
    response = await client.get("/recommendation")
    assert response.status_code == 200
    data = response.json()

    assert "recommendation_id" in data
    assert "optimal_price" in data
    assert "confidence_score" in data
    assert "requires_review" in data


@pytest.mark.asyncio
async def test_submit_override(client):
    """Test submitting override"""
    # First get recommendation
    rec_response = await client.get("/recommendation")
    assert rec_response.status_code == 200
    rec_data = rec_response.json()

    # Submit override
    override_data = {
        "recommendation_id": rec_data["recommendation_id"],
        "human_price": 18.50,
        "notes": "Test override",
        "reviewer_name": "Test User",
    }

    response = await client.post("/override", json=override_data)
    assert response.status_code == 200
    data = response.json()

    assert "override_id" in data
    assert data["human_price"] == 18.50
    assert data["status"] == "OVERRIDDEN"


@pytest.mark.asyncio
async def test_submit_override_invalid_id(client):
    """Test override with invalid recommendation ID"""
    override_data = {
        "recommendation_id": "invalid-id",
        "human_price": 18.50,
    }

    response = await client.post("/override", json=override_data)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_feedback(client):
    """Test getting feedback"""
    response = await client.get("/feedback")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "records" in data
