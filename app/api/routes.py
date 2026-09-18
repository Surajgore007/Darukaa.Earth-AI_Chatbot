"""
FastAPI route definitions for health checks and conversation.
"""

from fastapi import APIRouter, HTTPException
from app import __version__
from app.core.config import settings
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.graph.workflow import execute_chat_turn

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Returns system health status and configuration details.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        gemini_configured=settings.is_gemini_configured,
        database_configured=settings.is_database_configured,
        gemini_model=settings.gemini_model,
        embedding_model=settings.gemini_embedding_model
    )


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint handling multi-turn conversation, variable extraction,
    retrieval, reasoning, grounding validation, and confidence scoring.
    """
    try:
        result = execute_chat_turn(
            session_id=request.session_id,
            user_message=request.message
        )
        return ChatResponse(
            session_id=request.session_id,
            response=result.get("final_response", ""),
            is_complete=result.get("is_complete", False),
            confidence=result.get("confidence", "LOW"),
            variables=result.get("variables", {}),
            clarifying_question=result.get("clarifying_question")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing chat turn: {str(e)}")
