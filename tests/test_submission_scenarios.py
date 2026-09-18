"""
Comprehensive Submission Scenarios Validation Test (Scenarios A through F).
Directly tests the end-to-end challenge behaviors required for final submission readiness.
"""

import pytest
from app.graph.workflow import execute_chat_turn, SESSION_STORE
from app.graph.nodes.grounding_validator import validate_grounding
from app.graph.state import GraphState
from app.retrieval.retriever import DualLayerRetriever


def test_scenario_a_incomplete_query_causes_clarification():
    """
    Scenario A: Incomplete query
    'High synthetic fertilizer runoff is causing nitrate pollution in my local stream.'
    Expected: The system asks for useful missing environmental variables rather than
    immediately giving a generic recommendation.
    """
    session_id = "test-scenario-a"
    SESSION_STORE.pop(session_id, None)

    msg = "High synthetic fertilizer runoff is causing nitrate pollution in my local stream."
    res = execute_chat_turn(session_id, msg)

    assert res["is_complete"] is False, "Query with only pollution context must be recognized as incomplete"
    assert res["clarifying_question"] is not None
    # Must ask for missing core variables (soil carbon, rainfall, land use/crop)
    assert any(term in res["clarifying_question"].lower() for term in ["soil", "rainfall", "crop", "land use"])
    assert "RECOMMENDATION:" not in res["final_response"], "Should not give premature recommendation"


def test_scenario_b_complete_multi_variable_query():
    """
    Scenario B: Complete multi-variable query
    'My soil organic carbon is 0.3%, rainfall is low, I grow monoculture wheat in a semi-arid region, and there is nitrate pollution from fertilizer runoff.'
    Expected: The recommendation considers multiple variables together and retrieves both
    scientific context and quantitative metric evidence.
    """
    session_id = "test-scenario-b"
    SESSION_STORE.pop(session_id, None)

    msg = (
        "My soil organic carbon is 0.3%, rainfall is low, I grow monoculture wheat in a semi-arid region, "
        "and there is nitrate pollution from fertilizer runoff."
    )
    res = execute_chat_turn(session_id, msg)

    assert res["is_complete"] is True
    assert res["variables"]["soil_organic_carbon"] == "0.3%"
    assert res["variables"]["rainfall"] == "low"
    assert res["variables"]["crop_type"] == "wheat"
    assert "pollution" in res["variables"]

    response = res["final_response"]
    assert "RECOMMENDATION" in response
    assert "WHY IT WORKS" in response
    assert "IMPACTED METRICS" in response
    assert "SOURCES" in response
    assert len(res["retrieved_context"]["knowledge_chunks"]) > 0
    assert len(res["retrieved_context"]["metric_facts"]) > 0


def test_scenario_c_exact_unsupported_metric_query():
    """
    Scenario C: Exact unsupported metric
    'My farm has 0.3% SOC, low rainfall and monoculture wheat. By exactly what percentage will biodiversity increase from agroforestry?'
    Expected: Explicitly states retrieved evidence does not provide an exact biodiversity percentage.
    Must NOT substitute SOC or moisture numbers for biodiversity. Confidence must not be HIGH.
    """
    session_id = "test-scenario-c"
    SESSION_STORE.pop(session_id, None)

    msg = (
        "My farm has 0.3% SOC, low rainfall and monoculture wheat. "
        "By exactly what percentage will biodiversity increase from agroforestry?"
    )
    res = execute_chat_turn(session_id, msg)
    response = res["final_response"]

    assert (
        "exact percentage" in response.lower() or
        "exact figure" in response.lower() or
        "not provide an exact" in response.lower() or
        "cannot be reported" in response.lower()
    ), "Must explicitly state that exact biodiversity percentage is not in evidence"
    assert res["confidence"] != "HIGH", "Confidence must not be HIGH when requested exact metric is unevidenced"


def test_scenario_d_unsupported_numeric_claim_rejected():
    """
    Scenario D: Unsupported numeric claim
    Attempt to make the model produce a percentage that does not exist in retrieved evidence.
    Expected: Grounding validator rejects it.
    """
    state: GraphState = {
        "session_id": "test-scenario-d",
        "current_user_message": "test",
        "messages": [],
        "variables": {},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            "combined_context_prompt": "Verified Evidence: Legume cover crops improve SOC by +15-25% over 2-3 years."
        },
        # Fabricated number: 73.5%
        "draft_response": (
            "RECOMMENDATION:\n"
            "Apply miracle formula for 73.5% gain in soil quality.\n\n"
            "IMPACTED METRICS:\n"
            "- Soil quality: +73.5%\n\n"
            "SOURCES:\n"
            "- FAO (2020) | https://www.fao.org/documents/card/en/c/cb1928en"
        ),
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }

    res = validate_grounding(state)
    assert res["validation_passed"] is False, "Grounding validator must reject unverified 73.5%"
    assert res["retry_count"] == 1


def test_scenario_e_multi_turn_memory():
    """
    Scenario E: Multi-turn memory
    Provide environmental information across multiple messages.
    Expected: Variables from previous turns are preserved and used in final reasoning.
    """
    session_id = "test-scenario-e"
    SESSION_STORE.pop(session_id, None)

    # Turn 1: User mentions problem
    t1 = execute_chat_turn(session_id, "Biodiversity is declining on my farm.")
    assert t1["variables"]["biodiversity_indicators"] == "declining"

    # Turn 2: User provides SOC
    t2 = execute_chat_turn(session_id, "My soil organic carbon is 0.3%.")
    assert t2["variables"]["biodiversity_indicators"] == "declining"
    assert t2["variables"]["soil_organic_carbon"] == "0.3%"

    # Turn 3: User provides rainfall, crop, and region
    t3 = execute_chat_turn(session_id, "Rainfall is low and crop is monoculture wheat in a semi-arid zone.")
    assert t3["variables"]["biodiversity_indicators"] == "declining"
    assert t3["variables"]["soil_organic_carbon"] == "0.3%"
    assert t3["variables"]["rainfall"] == "low"
    assert t3["variables"]["crop_type"] == "wheat"
    assert t3["is_complete"] is True
    assert "RECOMMENDATION" in t3["final_response"]


def test_scenario_f_retrieval_separation():
    """
    Scenario F: Retrieval separation
    Verify that vector knowledge retrieval and structured metric-fact retrieval remain
    strictly separate in the code, data structures, and prompt blocks.
    """
    retrieval = DualLayerRetriever.retrieve(
        query="semi-arid wheat monoculture soil organic carbon 0.3% low rainfall",
        variable_tags=["soil", "climate", "land_use"],
        metric_keys=["soil_organic_carbon", "soil_moisture_retention"]
    )

    # 1. Structural separation in return dictionary
    assert "knowledge_chunks" in retrieval
    assert "metric_facts" in retrieval
    assert isinstance(retrieval["knowledge_chunks"], list)
    assert isinstance(retrieval["metric_facts"], list)

    # 2. Textual separation in combined prompt block
    prompt = retrieval["combined_context_prompt"]
    assert "GENERAL SCIENTIFIC CONTEXT:" in prompt
    assert "QUANTITATIVE FACTS:" in prompt
    
    # Ensure neither section is empty
    assert len(retrieval["general_scientific_context_str"]) > 50
    assert len(retrieval["quantitative_facts_str"]) > 50
