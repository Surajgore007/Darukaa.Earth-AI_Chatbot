"""
Pydantic API request and response schemas.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming chat request payload.
    """
    message: str = Field(..., description="User message or environmental query.", min_length=1)
    session_id: str = Field(default="default-session", description="Unique conversation session identifier.")


class ChatResponse(BaseModel):
    """
    Outgoing chat response payload.
    """
    session_id: str
    response: str
    is_complete: bool
    confidence: str
    variables: Dict[str, Any]
    clarifying_question: Optional[str] = None


class HealthResponse(BaseModel):
    """
    API Health check response.
    """
    status: str
    version: str
    gemini_configured: bool
    database_configured: bool
    gemini_model: str
    embedding_model: str
