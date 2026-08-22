"""Tests for review API"""

import pytest
from fastapi.testclient import TestClient

from pricepilot.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_root(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


def test_health(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_get_recommendation(client):
    """Test getting recommendation"""
    response = client.get("/recommendation")
    assert response.status_code == 200
    data = response.json()

    assert "recommendation_id" in data
    assert "optimal_price" in data
    assert "confidence_score" in data
    assert "requires_review" in data


def test_submit_override(client):
    """Test submitting override"""
    # First get recommendation
    rec_response = client.get("/recommendation")
    assert rec_response.status_code == 200
    rec_data = rec_response.json()

    # Submit override
    override_data = {
        "recommendation_id": rec_data["recommendation_id"],
        "human_price": 18.50,
        "notes": "Test override",
        "reviewer_name": "Test User",
    }

    response = client.post("/override", json=override_data)
    assert response.status_code == 200
    data = response.json()

    assert "override_id" in data
    assert data["human_price"] == 18.50
    assert data["status"] == "OVERRIDDEN"


def test_submit_override_invalid_id(client):
    """Test override with invalid recommendation ID"""
    override_data = {
        "recommendation_id": "invalid-id",
        "human_price": 18.50,
    }

    response = client.post("/override", json=override_data)
    assert response.status_code == 400


def test_get_feedback(client):
    """Test getting feedback"""
    response = client.get("/feedback")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "records" in data
