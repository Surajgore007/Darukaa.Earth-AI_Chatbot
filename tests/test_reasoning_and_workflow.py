"""
Unit tests for Phase 5: Reasoning, Grounding Validation, Confidence Scoring & Workflow.
"""

import pytest
from app.graph.workflow import execute_chat_turn, SESSION_STORE
from app.graph.nodes.grounding_validator import validate_grounding
from app.graph.nodes.confidence_scorer import compute_confidence
from app.graph.state import GraphState


def test_grounding_validator_catches_unsupported_number():
    """Verify validator flags unsupported numeric claims not in evidence."""
    state: GraphState = {
        "session_id": "test-session",
        "current_user_message": "test",
        "messages": [],
        "variables": {},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            "combined_context_prompt": "Evidence: legume cover crops increase SOC by +15–25% over 2–3 years (FAO)."
        },
        # Hallucinated number: 45% (not in evidence!)
        "draft_response": "RECOMMENDATION: legume cover crops.\nIMPACT: 45% improvement over 2–3 years.",
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }
    res = validate_grounding(state)
    # Should fail validation and trigger retry
    assert res["validation_passed"] is False
    assert res["retry_count"] == 1

    # Second failure should yield challenge-mandated fallback
    state["retry_count"] = 1
    res2 = validate_grounding(state)
    assert res2["validation_passed"] is True
    assert "Insufficient evidence for a confident recommendation." in res2["final_response"]


def test_confidence_scorer_high_when_metric_fact_matches():
    """Verify confidence is scored as HIGH when matching metric_facts exists."""
    state: GraphState = {
        "session_id": "test-session",
        "current_user_message": "test",
        "messages": [],
        "variables": {},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            "metric_facts": [
                {
                    "intervention": "legume-based cover crops",
                    "affects_metric": "soil_organic_carbon",
                    "effect_value": "+15–25% over 2–3 years"
                }
            ],
            "knowledge_chunks": []
        },
        "draft_response": "RECOMMENDATION:\nUse legume-based cover crops to increase soil organic carbon.\n\nCONFIDENCE:\n[CALCULATED_BY_SYSTEM]",
        "validation_passed": True,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": "RECOMMENDATION:\nUse legume-based cover crops to increase soil organic carbon.\n\nCONFIDENCE:\n[CALCULATED_BY_SYSTEM]"
    }
    res = compute_confidence(state)
    assert res["confidence"] == "HIGH"
    assert "CONFIDENCE:\nHIGH" in res["final_response"]


def test_full_workflow_multi_turn():
    """Test full multi-turn conversation flow from clarification to grounded recommendation."""
    session_id = "test-flow-123"
    SESSION_STORE.pop(session_id, None)

    # Turn 1: User gives incomplete input
    res1 = execute_chat_turn(session_id, "Biodiversity is declining on my land.")
    assert res1["is_complete"] is False
    assert "clarifying_question" in res1 and res1["clarifying_question"] is not None
    assert "soil organic carbon" in res1["final_response"]

    # Turn 2: User provides environmental context
    res2 = execute_chat_turn(
        session_id,
        "My soil organic carbon is 0.3%, rainfall is low, and crop is monoculture wheat in a semi-arid region."
    )
    assert res2["is_complete"] is True
    assert "RECOMMENDATION:" in res2["final_response"]
    assert "WHY IT WORKS:" in res2["final_response"]
    assert "IMPACTED METRICS:" in res2["final_response"]
    assert "TIME HORIZON:" in res2["final_response"]
    assert "CONFIDENCE:" in res2["final_response"]
    assert "SOURCES:" in res2["final_response"]
    assert res2["confidence"] in {"HIGH", "MEDIUM"}
