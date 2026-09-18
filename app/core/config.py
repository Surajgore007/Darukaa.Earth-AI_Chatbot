"""
Application Configuration Module.

This module centralizes all environment variables and configuration settings
using Pydantic Settings. This ensures:
1. Environment variables are typed and validated.
2. Default values are clearly documented and safe.
3. Secrets are read from .env without being hard-coded in code.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings loaded from environment variables or a .env file.
    """

    # --------------------------------------------------------------------------
    # Google Gemini API Settings
    # --------------------------------------------------------------------------
    # Google Gemini API Key. Obtained from https://aistudio.google.com/
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        description="Google Gemini API Key for LLM reasoning and embeddings."
    )

    # Currently supported Google Gemini generation model (Verified September 2026: gemini-2.5-flash)
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="GEMINI_MODEL",
        description="Active Google Gemini model for multi-metric reasoning."
    )

    # Currently supported Google Gemini embedding model (Verified September 2026: gemini-embedding-001)
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias="GEMINI_EMBEDDING_MODEL",
        description="Active Google Gemini embedding model for semantic vector search."
    )

    # --------------------------------------------------------------------------
    # Groq API Settings (Fast, high-rate-limit fallback provider)
    # --------------------------------------------------------------------------
    groq_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GROQ_API_KEY",
        description="Groq API Key for high-speed LLM reasoning."
    )

    groq_model: str = Field(
        default="qwen/qwen3.8-27b",
        validation_alias="GROQ_MODEL",
        description="Active Groq model for multi-metric reasoning."
    )

    # --------------------------------------------------------------------------
    # Supabase PostgreSQL Settings
    # --------------------------------------------------------------------------
    # Supabase PostgreSQL connection URI with pgvector extension
    # Example: postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
    database_url: Optional[str] = Field(
        default=None,
        validation_alias="DATABASE_URL",
        description="PostgreSQL connection URI for Supabase."
    )

    # --------------------------------------------------------------------------
    # Application & Environment Settings
    # --------------------------------------------------------------------------
    environment: str = Field(
        default="development",
        validation_alias="ENVIRONMENT",
        description="Runtime environment: 'development', 'staging', or 'production'."
    )

    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging verbosity level (DEBUG, INFO, WARNING, ERROR)."
    )

    app_host: str = Field(
        default="127.0.0.1",
        validation_alias="APP_HOST",
        description="FastAPI bind host."
    )

    app_port: int = Field(
        default=8000,
        validation_alias="APP_PORT",
        description="FastAPI bind port."
    )

    # Pydantic configuration: load from .env with UTF-8 encoding
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def is_gemini_configured(self) -> bool:
        """Check if a Gemini API key is provided and not empty."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def is_groq_configured(self) -> bool:
        """Check if a Groq API key is provided and not empty."""
        return bool(self.groq_api_key and self.groq_api_key.strip())

    @property
    def is_database_configured(self) -> bool:
        """Check if a Database URL is provided and not empty."""
        return bool(self.database_url and self.database_url.strip())


# Create a singleton settings instance for use across the application
settings = Settings()
