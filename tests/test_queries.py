"""
Comprehensive Challenge Evaluation Test Suite (test_queries.py).

Covers all 9 challenge-mandated scenarios:
1. Complete environmental query
2. Multi-turn conversation
3. Missing-variable clarification
4. Soil + rainfall + land-use reasoning
5. Biodiversity-focused question
6. Human-impact question
7. Quantitative evidence retrieval
8. Query with insufficient scientific evidence
9. Grounding-validator failure case
"""

import sys
import os
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.workflow import execute_chat_turn, SESSION_STORE
from app.graph.nodes.grounding_validator import validate_grounding
from app.graph.nodes.confidence_scorer import compute_confidence
from app.retrieval.structured_search import search_metric_facts
from app.retrieval.vector_search import search_knowledge_chunks
from app.graph.state import GraphState


def test_1_complete_environmental_query():
    """Scenario 1: Complete environmental query with multiple variables."""
    session_id = "test-scenario-1"
    SESSION_STORE.pop(session_id, None)

    message = "soil organic carbon = 0.3%, rainfall = low, crop = monoculture wheat, region = semi-arid"
    result = execute_chat_turn(session_id, message)

    assert result["is_complete"] is True
    assert result["variables"]["soil_organic_carbon"] == "0.3%"
    assert result["variables"]["rainfall"] == "low"
    assert result["variables"]["crop_type"] == "wheat"

    response = result["final_response"]
    assert "RECOMMENDATION" in response
    assert "WHY IT WORKS" in response
    assert "IMPACTED METRICS" in response
    assert "SOURCES" in response
    assert result["confidence"] in {"HIGH", "MEDIUM"}


def test_2_multi_turn_conversation():
    """Scenario 2: Multi-turn conversation with state memory across turns."""
    session_id = "test-scenario-2"
    SESSION_STORE.pop(session_id, None)

    # Turn 1: Broad problem statement
    t1 = execute_chat_turn(session_id, "Biodiversity is declining on my farm.")
    assert t1["is_complete"] is False
    assert t1["clarifying_question"] is not None
    assert t1["variables"]["biodiversity_indicators"] == "declining"

    # Turn 2: Supplies soil organic carbon
    t2 = execute_chat_turn(session_id, "My soil organic carbon is 0.3%.")
    # State retains Turn 1 variable
    assert t2["variables"]["biodiversity_indicators"] == "declining"
    assert t2["variables"]["soil_organic_carbon"] == "0.3%"

    # Turn 3: Supplies rainfall and land use
    t3 = execute_chat_turn(session_id, "Rainfall is low and we grow monoculture wheat in a semi-arid zone.")
    assert t3["is_complete"] is True
    assert t3["variables"]["rainfall"] == "low"
    assert t3["variables"]["crop_type"] == "wheat"
    assert "RECOMMENDATION" in t3["final_response"]


def test_3_missing_variable_clarification():
    """Scenario 3: System identifies missing context and asks single focused question."""
    session_id = "test-scenario-3"
    SESSION_STORE.pop(session_id, None)

    result = execute_chat_turn(session_id, "How can I restore my land?")
    assert result["is_complete"] is False
    assert result["clarifying_question"] is not None
    assert "soil organic carbon" in result["clarifying_question"]
    assert "rainfall" in result["clarifying_question"]


def test_4_multi_variable_reasoning():
    """Scenario 4: System connects Soil + Rainfall + Land use together in reasoning."""
    session_id = "test-scenario-4"
    SESSION_STORE.pop(session_id, None)

    message = "soil organic carbon: 0.3%, rainfall: low, crop: monoculture wheat"
    result = execute_chat_turn(session_id, message)

    response = result["final_response"]
    # Check that reasoning mentions the interplay of carbon, moisture/rainfall, and monoculture
    assert "0.3%" in response or "carbon" in response.lower()
    assert "moisture" in response.lower() or "drought" in response.lower() or "rainfall" in response.lower()
    assert "monoculture" in response.lower() or "diversif" in response.lower()


def test_5_biodiversity_focused_question():
    """Scenario 5: Biodiversity decline with wild pollinators."""
    session_id = "test-scenario-5"
    SESSION_STORE.pop(session_id, None)

    message = "We have low pollinators and fewer bees in our apple orchard. Soil pH is 6.5 and rainfall is moderate."
    result = execute_chat_turn(session_id, message)

    assert result["is_complete"] is True
    response = result["final_response"]
    assert "hedgerow" in response.lower() or "flowering" in response.lower() or "pollinator" in response.lower()
    assert "IPBES" in response or "pollinator" in response.lower()


def test_6_human_impact_question():
    """Scenario 6: Human impact query addressing synthetic fertilizer and nitrate runoff."""
    session_id = "test-scenario-6"
    SESSION_STORE.pop(session_id, None)

    message = "High synthetic fertilizer runoff and pesticide wash is polluting our river. Land use is corn cropland."
    result = execute_chat_turn(session_id, message)

    assert result["is_complete"] is True
    response = result["final_response"]
    assert "riparian" in response.lower() or "buffer" in response.lower() or "residue" in response.lower()


def test_7_quantitative_evidence_retrieval():
    """Scenario 7: Directly verify quantitative evidence retrieval from metric_facts."""
    facts = search_metric_facts(metrics=["soil_organic_carbon"])
    assert len(facts) > 0
    fact = facts[0]
    assert fact["affects_metric"] == "soil_organic_carbon"
    assert "+15–25%" in fact["effect_value"]
    assert "FAO" in fact["source"]


def test_8_insufficient_scientific_evidence():
    """Scenario 8: Query with unsupported scientific concepts produces low confidence / disclaimer."""
    session_id = "test-scenario-8"
    SESSION_STORE.pop(session_id, None)

    # Completely alien query with no matching scientific basis in FAO/IPCC
    unknown_query = "Can we use quantum crystal telepathy to increase crop yield? region: unknown"
    result = execute_chat_turn(session_id, unknown_query)

    # Should ask for clarification or produce LOW confidence
    assert result["confidence"] == "LOW" or result["is_complete"] is False


def test_9_grounding_validator_failure_case():
    """Scenario 9: Grounding validator detects hallucinated number and returns challenge-mandated failure."""
    state: GraphState = {
        "session_id": "test-scenario-9",
        "current_user_message": "test",
        "messages": [],
        "variables": {},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            "combined_context_prompt": "Verified Evidence: Cover crops increase SOC by +15–25% over 2–3 years (FAO)."
        },
        # Fabricated number: +95% (not in evidence!)
        "draft_response": "RECOMMENDATION: Use miracle treatment for +95% carbon in 10 days.",
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }

    # Attempt 1 -> Fails and triggers retry
    res1 = validate_grounding(state)
    assert res1["validation_passed"] is False
    assert res1["retry_count"] == 1

    # Attempt 2 -> After second failure, returns exact challenge phrase
    state["retry_count"] = 1
    res2 = validate_grounding(state)
    assert res2["validation_passed"] is True
    assert res2["final_response"] == "Insufficient evidence for a confident recommendation."


def run_all_tests():
    """CLI runner to execute and report all 9 scenarios."""
    print("\n" + "=" * 60)
    print("Darukaa.Earth Biodiversity Chatbot — 9 Benchmark Tests")
    print("=" * 60)
    tests = [
        ("Test 1: Complete Environmental Query", test_1_complete_environmental_query),
        ("Test 2: Multi-Turn Conversation", test_2_multi_turn_conversation),
        ("Test 3: Missing-Variable Clarification", test_3_missing_variable_clarification),
        ("Test 4: Multi-Variable Reasoning", test_4_multi_variable_reasoning),
        ("Test 5: Biodiversity-Focused Question", test_5_biodiversity_focused_question),
        ("Test 6: Human-Impact Question", test_6_human_impact_question),
        ("Test 7: Quantitative Evidence Retrieval", test_7_quantitative_evidence_retrieval),
        ("Test 8: Insufficient Scientific Evidence", test_8_insufficient_scientific_evidence),
        ("Test 9: Grounding-Validator Failure Case", test_9_grounding_validator_failure_case),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            test_func()
            print(f"  [PASSED] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAILED] {name}: {e}")

    print("-" * 60)
    print(f"Result: {passed}/{len(tests)} benchmark scenarios passed.")
    print("=" * 60 + "\n")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
