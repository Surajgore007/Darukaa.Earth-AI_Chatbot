"""
Unit tests for Phase 4: LangGraph State, Variable Extraction & Completeness Check.
"""

import pytest
from app.graph.nodes.extract_variables import extract_variables
from app.graph.nodes.check_completeness import check_completeness
from app.graph.state import GraphState


def test_extract_variables_multi_turn_preservation():
    """Verify variables from Turn 1 are preserved when Turn 2 variables arrive."""
    # Turn 1
    state_turn1: GraphState = {
        "session_id": "test-session",
        "current_user_message": "Biodiversity is declining on my land.",
        "messages": [],
        "variables": {},
        "is_complete": False,
        "clarifying_question": None,
        "retrieved_context": {},
        "draft_response": None,
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }
    res1 = extract_variables(state_turn1)
    assert res1["variables"]["biodiversity_indicators"] == "declining"

    # Turn 2: User supplies SOC
    state_turn2: GraphState = {
        **state_turn1,
        "current_user_message": "My soil organic carbon is 0.3%.",
        "variables": res1["variables"]
    }
    res2 = extract_variables(state_turn2)
    
    # Both Turn 1 and Turn 2 variables must now be present
    assert res2["variables"]["biodiversity_indicators"] == "declining"
    assert res2["variables"]["soil_organic_carbon"] == "0.3%"


def test_extract_variables_full_query():
    """Verify extraction of multiple variables from a single structured query."""
    state: GraphState = {
        "session_id": "test-session",
        "current_user_message": "soil organic carbon = 0.3%, rainfall = low, crop = monoculture wheat, region = semi-arid",
        "messages": [],
        "variables": {},
        "is_complete": False,
        "clarifying_question": None,
        "retrieved_context": {},
        "draft_response": None,
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }
    res = extract_variables(state)
    vars_dict = res["variables"]
    
    assert vars_dict.get("soil_organic_carbon") == "0.3%"
    assert vars_dict.get("rainfall") == "low"
    assert vars_dict.get("land_use") == "monoculture"
    assert vars_dict.get("crop_type") == "wheat"
    assert vars_dict.get("region") == "semi-arid"


def test_check_completeness_incomplete():
    """Verify system detects missing variables and asks a single focused question."""
    state: GraphState = {
        "session_id": "test-session",
        "current_user_message": "Biodiversity is declining on my farm.",
        "messages": [],
        "variables": {"biodiversity_indicators": "declining"},
        "is_complete": False,
        "clarifying_question": None,
        "retrieved_context": {},
        "draft_response": None,
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }
    res = check_completeness(state)
    assert res["is_complete"] is False
    assert res["clarifying_question"] is not None
    assert "soil organic carbon" in res["clarifying_question"]
    assert "rainfall" in res["clarifying_question"]


def test_check_completeness_sufficient():
    """Verify completeness passes when sufficient context variables exist."""
    state: GraphState = {
        "session_id": "test-session",
        "current_user_message": "Here is my data.",
        "messages": [],
        "variables": {
            "soil_organic_carbon": "0.3%",
            "rainfall": "low",
            "crop_type": "wheat",
            "region": "semi-arid"
        },
        "is_complete": False,
        "clarifying_question": None,
        "retrieved_context": {},
        "draft_response": None,
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }
    res = check_completeness(state)
    assert res["is_complete"] is True
    assert res["clarifying_question"] is None
