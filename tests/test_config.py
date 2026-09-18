"""
Unit tests for configuration and environment handling.
"""

import pytest
from app.core.config import Settings


def test_default_settings_models():
    """Verify that default models are current and supported."""
    settings = Settings()
    # Check that generation model is gemini-2.5-flash
    assert settings.gemini_model == "gemini-2.5-flash"
    assert "2.0" not in settings.gemini_model  # ensure not deprecated 2.0

    # Check embedding model is gemini-embedding-001
    assert settings.gemini_embedding_model == "gemini-embedding-001"
    assert "004" not in settings.gemini_embedding_model  # ensure not retired text-embedding-004


def test_settings_properties():
    """Verify boolean status helpers for credentials."""
    empty_settings = Settings(gemini_api_key=None, database_url=None)
    assert empty_settings.is_gemini_configured is False
    assert empty_settings.is_database_configured is False

    configured_settings = Settings(
        gemini_api_key="test-key-123",
        database_url="postgresql://user:pass@localhost:5432/db"
    )
    assert configured_settings.is_gemini_configured is True
    assert configured_settings.is_database_configured is True
