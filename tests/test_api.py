"""
Integration tests for FastAPI endpoints (/health, /chat, /).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 and accurate system configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["gemini_model"] == "gemini-2.5-flash"
    assert data["embedding_model"] == "gemini-embedding-001"


def test_frontend_served():
    """Verify GET / returns 200 and serves HTML interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Darukaa.Earth" in response.text
    assert "chat-container" in response.text


def test_chat_endpoint_incomplete_input():
    """Verify /chat asks clarifying question when context is missing."""
    payload = {
        "message": "Biodiversity is declining on my farm.",
        "session_id": "api-test-session-1"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_complete"] is False
    assert "soil organic carbon" in data["response"]
    assert data["confidence"] == "LOW"


def test_chat_endpoint_complete_input():
    """Verify /chat produces grounded recommendation for complete query."""
    payload = {
        "message": "soil organic carbon = 0.3%, rainfall = low, crop = monoculture wheat, region = semi-arid",
        "session_id": "api-test-session-2"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_complete"] is True
    assert "RECOMMENDATION:" in data["response"]
    assert "WHY IT WORKS:" in data["response"]
    assert "SOURCES:" in data["response"]
    assert data["confidence"] in {"HIGH", "MEDIUM"}
    assert data["variables"]["soil_organic_carbon"] == "0.3%"
    assert data["variables"]["rainfall"] == "low"
