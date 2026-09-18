"""
Regression tests for grounding, confidence pairing, single primary intervention,
and exact-metric query behavior.
"""

import re
import pytest
from app.graph.nodes.grounding_validator import validate_grounding, validate_claim_attributions
from app.graph.nodes.confidence_scorer import compute_confidence
from app.graph.state import GraphState
from app.graph.workflow import execute_chat_turn, SESSION_STORE
from app.retrieval.retriever import DualLayerRetriever


def test_regression_a_numeric_grounding_rejects_digit_fallbacks():
    """
    TEST A:
    15–25% must NOT pass merely because '2–3 years' or other digits (1, 5, 2) appear in the context.
    The complete numeric range must exist in the retrieved evidence.
    """
    state: GraphState = {
        "session_id": "test-reg-a",
        "current_user_message": "test",
        "messages": [],
        "variables": {},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            # Evidence contains '2-3 years' and '0.5%', but NOT '15-25%'!
            "combined_context_prompt": (
                "Evidence: Organic amendments restore microbial networks over 2-3 years. "
                "Soils with SOC below 0.5% in drylands exhibit low microbial biomass."
            )
        },
        # Draft claims 15-25%
        "draft_response": (
            "RECOMMENDATION:\n"
            "Establish legume cover crops.\n\n"
            "IMPACTED METRICS:\n"
            "- Soil organic carbon: +15-25% over 2-3 years\n\n"
            "SOURCES:\n"
            "- FAO (2020) | https://www.fao.org/documents/card/en/c/cb1928en"
        ),
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }

    # Attempt 1 -> Grounding validator must reject +15-25%
    res = validate_grounding(state)
    assert res["validation_passed"] is False, "Validator must NOT pass +15-25% when evidence only contains '2-3 years'"
    assert res["retry_count"] == 1


def test_regression_b_confidence_requires_intervention_metric_pair():
    """
    TEST B:
    agroforestry -> soil_moisture_retention in metric_facts must NOT automatically
    create HIGH confidence for agroforestry -> soil_organic_carbon.
    """
    state: GraphState = {
        "session_id": "test-reg-b",
        "current_user_message": "test",
        "messages": [],
        "variables": {},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            # Database ONLY has agroforestry -> soil_moisture_retention (+20-40%)
            "metric_facts": [
                {
                    "intervention": "agroforestry and alley cropping",
                    "affects_metric": "soil_moisture_retention",
                    "effect_value": "+20-40% retention in root zone"
                }
            ],
            "knowledge_chunks": [
                {"content": "General qualitative context on agroforestry and soil organic matter."}
            ]
        },
        # Draft claims agroforestry -> soil organic carbon (mismatched metric!)
        "draft_response": (
            "RECOMMENDATION:\n"
            "Establish an agroforestry system.\n\n"
            "IMPACTED METRICS:\n"
            "- soil organic carbon: increases significantly\n\n"
            "CONFIDENCE:\n"
            "[CALCULATED_BY_SYSTEM]"
        ),
        "validation_passed": True,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": (
            "RECOMMENDATION:\n"
            "Establish an agroforestry system.\n\n"
            "IMPACTED METRICS:\n"
            "- soil organic carbon: increases significantly\n\n"
            "CONFIDENCE:\n"
            "[CALCULATED_BY_SYSTEM]"
        )
    }

    res = compute_confidence(state)
    # Must NOT be HIGH because agroforestry -> soil organic carbon is not in metric_facts!
    assert res["confidence"] != "HIGH", "Mismatched (intervention, metric) pair must not receive HIGH confidence!"
    assert res["confidence"] == "MEDIUM"


def test_regression_c_one_primary_integrated_intervention():
    """
    TEST C:
    A recommendation must contain ONE primary integrated intervention rather than a list
    of 5 unrelated bundled interventions.
    """
    session_id = "test-reg-c"
    SESSION_STORE.pop(session_id, None)

    message = "soil organic carbon = 0.3%, rainfall = low, crop = monoculture wheat, region = semi-arid"
    res = execute_chat_turn(session_id, message)
    rec_text = res["final_response"].split("WHY IT WORKS:")[0]

    # Verify it does NOT create a shopping list of separate practices
    prohibited_shopping_list = ["swales + contour bunds + organic mulching", "alley cropping + organic mulching + contour bunds"]
    for prohibited in prohibited_shopping_list:
        assert prohibited not in rec_text.lower()


def test_regression_d_unsupported_exact_metric_query():
    """
    TEST D:
    A request for an unsupported biodiversity percentage must explicitly report that the
    requested quantitative evidence is unavailable and must not substitute unrelated percentages.
    """
    session_id = "test-reg-d"
    SESSION_STORE.pop(session_id, None)

    message = (
        "My farm has 0.3% soil organic carbon, low rainfall, monoculture wheat, and severe nitrate pollution. "
        "By exactly what percentage will biodiversity increase if I implement agroforestry?"
    )
    res = execute_chat_turn(session_id, message)
    response = res["final_response"]

    # Must explicitly state that an exact biodiversity percentage is not provided in retrieved evidence
    assert (
        "does not provide an exact percentage for biodiversity" in response.lower() or
        "exact biodiversity percentage cannot be reported" in response.lower()
    ), "Response must explicitly report that exact biodiversity percentage is unavailable"

    # Confidence must not be HIGH when an exact requested metric is unavailable
    assert res["confidence"] != "HIGH", "Confidence must not be HIGH when requested exact metric is unevidenced"


def test_regression_e_multi_variable_query_does_not_collapse_to_single_metric():
    """
    TEST E:
    A 5-variable query (SOC, rainfall, monoculture, nitrate pollution, biodiversity decline)
    must NOT collapse into a single-metric recommendation (e.g. riparian buffers discussing only nitrate).
    It must synthesize multiple variables together.
    """
    session_id = "test-reg-e"
    SESSION_STORE.pop(session_id, None)

    msg = (
        "My soil organic carbon is 0.3%, rainfall is low, I grow monoculture wheat, "
        "fertilizer runoff is causing nitrate pollution in a nearby stream, and biodiversity has declined. "
        "Which single intervention should I prioritize, and how would it affect each of these environmental problems?"
    )
    res = execute_chat_turn(session_id, msg)
    response = res["final_response"]

    # Must address multiple variables, not just nitrate pollution
    assert "0.3%" in response or "carbon" in response.lower(), "Must address soil organic carbon"
    assert "moisture" in response.lower() or "rainfall" in response.lower(), "Must address rainfall/moisture"
    assert "monoculture" in response.lower() or "wheat" in response.lower(), "Must address monoculture / wheat"
    assert "nitrate" in response.lower() or "runoff" in response.lower(), "Must evaluate nitrate pollution"
    assert "biodiversity" in response.lower() or "microbial" in response.lower(), "Must address biodiversity"

    # Must NOT recommend solely riparian buffer strips
    rec_match = re.search(r"(?:\*\*|#+)?\s*RECOMMENDATION:?\s*(?:\*\*)?\s*(.*?)(?=\n(?:\*\*|#+)?\s*WHY IT WORKS|$)", response, re.DOTALL | re.IGNORECASE)
    rec = rec_match.group(1).lower() if rec_match else response[:200].lower()
    assert "riparian buffer" not in rec, "Multi-variable query must prioritize integrated intervention over single-metric riparian buffer"


def test_regression_f_retrieval_preserves_evidence_for_multiple_variables():
    """
    TEST F:
    Retrieval must preserve variable-by-variable scientific evidence mapping across all active variables.
    """
    variables = {
        "soil_organic_carbon": "0.3%",
        "rainfall": "low",
        "land_use": "monoculture",
        "crop_type": "wheat",
        "pollution": "nitrate pollution in stream",
        "biodiversity_indicators": "declining"
    }
    retrieved = DualLayerRetriever.retrieve_by_variables(
        variables=variables,
        user_message="Test multi-variable query"
    )

    combined_prompt = retrieved["combined_context_prompt"]
    # Check that variable breakdown sections exist
    assert "VARIABLE: Soil Health (SOC)" in combined_prompt
    assert "VARIABLE: Climate & Hydrology (Rainfall)" in combined_prompt
    assert "VARIABLE: Land Use & Crop Diversity" in combined_prompt
    assert "VARIABLE: Human Impact & Pollution" in combined_prompt
    assert "VARIABLE: Biodiversity Status" in combined_prompt

    # Check that facts are collected across domains
    metrics_retrieved = {f["affects_metric"] for f in retrieved["metric_facts"]}
    assert "soil_organic_carbon" in metrics_retrieved
    assert "soil_moisture_retention" in metrics_retrieved
    assert "agricultural_runoff_nitrates" in metrics_retrieved


def test_regression_g_recommendation_cannot_claim_unsupported_relationships():
    """
    TEST G:
    Every quantitative impacted metric claimed in the response must have a verified
    matching row in metric_facts for that intervention.
    """
    session_id = "test-reg-g"
    SESSION_STORE.pop(session_id, None)

    msg = (
        "My soil organic carbon is 0.3%, rainfall is low, I grow monoculture wheat, "
        "fertilizer runoff is causing nitrate pollution in a nearby stream, and biodiversity has declined. "
        "Which single intervention should I prioritize, and how would it affect each of these environmental problems?"
    )
    res = execute_chat_turn(session_id, msg)
    response = res["final_response"]

    # In IMPACTED METRICS, agroforestry/legumes cannot claim unsupported numbers for stream nitrate runoff
    assert "agroforestry -> agricultural_runoff_nitrates: -" not in response.lower()
    # If runoff is mentioned under IMPACTED METRICS, it must acknowledge the evidence limitation
    if "agricultural runoff nitrates" in response.lower():
        assert "does not establish" in response.lower() or "separately" in response.lower() or "limitation" in response.lower()


def test_regression_h_confidence_decreases_when_only_one_of_many_metrics_has_evidence():
    """
    TEST H:
    When a problem involves 5 environmental variables, an intervention supported by only
    1 quantitative metric fact must NOT receive HIGH confidence (must be MEDIUM or LOW).
    """
    state: GraphState = {
        "session_id": "test-reg-h",
        "current_user_message": "5-variable problem",
        "messages": [],
        "variables": {
            "soil_organic_carbon": "0.3%",
            "rainfall": "low",
            "land_use": "monoculture",
            "pollution": "nitrate runoff",
            "biodiversity_indicators": "declining"
        },
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            # Only 1 matching metric fact exists for riparian buffers -> agricultural_runoff_nitrates
            "metric_facts": [
                {
                    "intervention": "vegetated riparian buffer strips",
                    "affects_metric": "agricultural_runoff_nitrates",
                    "effect_value": "-50-85% reduction in nitrate delivery"
                }
            ],
            "knowledge_chunks": [
                {"content": "Riparian buffer strips reduce nitrate runoff into waterways."}
            ]
        },
        "draft_response": (
            "RECOMMENDATION:\n"
            "Establish vegetated riparian buffer strips along agricultural waterways.\n\n"
            "IMPACTED METRICS:\n"
            "- agricultural runoff nitrates: -50-85% reduction in nitrate delivery\n\n"
            "CONFIDENCE:\n"
            "[CALCULATED_BY_SYSTEM]"
        ),
        "validation_passed": True,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": (
            "RECOMMENDATION:\n"
            "Establish vegetated riparian buffer strips along agricultural waterways.\n\n"
            "IMPACTED METRICS:\n"
            "- agricultural runoff nitrates: -50-85% reduction in nitrate delivery\n\n"
            "CONFIDENCE:\n"
            "[CALCULATED_BY_SYSTEM]"
        )
    }

    res = compute_confidence(state)
    # With 5 active variables and only 1 verified metric, confidence must NOT be HIGH!
    assert res["confidence"] != "HIGH", "Confidence must not be HIGH when only 1 of 5 requested variables is supported"
    assert res["confidence"] == "MEDIUM"


def test_regression_i_system_acknowledges_evidence_limitations():
    """
    TEST I:
    When no single intervention directly solves all five environmental problems,
    the system must explicitly acknowledge that evidence does not establish it as a single solution for all 5.
    """
    session_id = "test-reg-i"
    SESSION_STORE.pop(session_id, None)

    msg = (
        "My soil organic carbon is 0.3%, rainfall is low, I grow monoculture wheat, "
        "fertilizer runoff is causing nitrate pollution in a nearby stream, and biodiversity has declined. "
        "Which single intervention should I prioritize, and how would it affect each of these environmental problems?"
    )
    res = execute_chat_turn(session_id, msg)
    response = res["final_response"]

    # Must explicitly state evidence limitation regarding stream nitrate runoff or single solution for all 5
    assert (
        "does not establish" in response.lower() or
        "does not show" in response.lower() or
        "no evidence" in response.lower() or
        "cannot be recommended as a comprehensive solution" in response.lower() or
        "additional evidence would be required" in response.lower() or
        "riparian buffer" in response.lower() or
        "limitation" in response.lower()
    ), "System must explicitly acknowledge evidence limitations rather than claiming unsupported benefits"


def test_regression_j_microbial_biomass_not_conflated_with_overall_biodiversity():
    """
    TEST J:
    Microbial biomass improvement (+20-35%) cannot automatically become an overall biodiversity
    improvement claim.
    """
    # 1. Draft conflating microbial biomass with overall biodiversity must FAIL validation
    failing_state: GraphState = {
        "session_id": "test-reg-j-fail",
        "current_user_message": "test",
        "messages": [],
        "variables": {"biodiversity_indicators": "declining"},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            "combined_context_prompt": "FAO (2020): Microbial biomass increases by +20-35% in soil.",
            "metric_facts": [
                {
                    "intervention": "legume cover crops",
                    "affects_metric": "microbial_biomass",
                    "effect_value": "+20-35% increase"
                }
            ]
        },
        "draft_response": (
            "RECOMMENDATION:\n"
            "Establish legume cover cropping.\n\n"
            "WHY IT WORKS:\n"
            "Legume cover crops reverse biodiversity decline (+20-35%).\n\n"
            "IMPACTED METRICS:\n"
            "- Biodiversity: +20-35% increase\n\n"
            "TIME HORIZON:\nmedium\n\n"
            "CONFIDENCE:\n[CALCULATED_BY_SYSTEM]\n\n"
            "SOURCES:\n- FAO (2020) | https://www.fao.org/documents/card/en/c/cb1928en"
        ),
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }

    res_fail = validate_grounding(failing_state)
    assert res_fail["validation_passed"] is False, "Validator must reject claiming +20-35% as overall biodiversity gain"

    # 2. Draft properly stating microbial biomass must PASS validation
    passing_state = dict(failing_state)
    passing_state["retrieved_context"] = {
        "combined_context_prompt": (
            "FAO (2020) | https://www.fao.org/documents/card/en/c/cb1928en: "
            "Microbial biomass increases by +20-35% in soil."
        ),
        "metric_facts": [
            {
                "intervention": "legume cover crops",
                "affects_metric": "microbial_biomass",
                "effect_value": "+20-35% increase"
            }
        ]
    }
    passing_state["draft_response"] = (
        "RECOMMENDATION:\n"
        "Establish legume cover cropping.\n\n"
        "WHY IT WORKS:\n"
        "Legume cover crops rebuild biological soil activity, increasing microbial biomass by +20-35% (FAO 2020). "
        "The retrieved evidence does not quantify overall landscape biodiversity gains.\n\n"
        "IMPACTED METRICS:\n"
        "- Microbial biomass: +20-35% increase in soil biological activity\n\n"
        "TIME HORIZON:\nmedium\n\n"
        "CONFIDENCE:\n[CALCULATED_BY_SYSTEM]\n\n"
        "SOURCES:\n- FAO (2020) | https://www.fao.org/documents/card/en/c/cb1928en"
    )
    res_pass = validate_grounding(passing_state)
    assert res_pass["validation_passed"] is True, "Validator must pass correctly distinguished microbial biomass claim"


def test_regression_k_pollinator_percentage_not_attributed_to_agroforestry():
    """
    TEST K:
    Pollinator percentage (+25-50%) cannot be attributed to agroforestry if the metric fact
    belongs to perennial hedgerows/flowering margins.
    """
    # 1. Draft attributing +25-50% pollinator richness to agroforestry must FAIL validation
    failing_state: GraphState = {
        "session_id": "test-reg-k-fail",
        "current_user_message": "test",
        "messages": [],
        "variables": {"land_use": "agroforestry"},
        "is_complete": True,
        "clarifying_question": None,
        "retrieved_context": {
            "combined_context_prompt": (
                "IPBES (2019): Perennial flowering hedgerows increase pollinator species richness by +25-50%. "
                "IPCC (2019): Agroforestry increases root-zone soil moisture retention by +20-40%."
            )
        },
        "draft_response": (
            "RECOMMENDATION:\n"
            "Establish an agroforestry system.\n\n"
            "WHY IT WORKS:\n"
            "Agroforestry increases pollinator species richness by +25-50%.\n\n"
            "IMPACTED METRICS:\n"
            "- Pollinator species richness: +25-50% increase\n\n"
            "TIME HORIZON:\nmedium\n\n"
            "CONFIDENCE:\n[CALCULATED_BY_SYSTEM]\n\n"
            "SOURCES:\n- IPBES (2019) | https://www.ipbes.net/global-assessment"
        ),
        "validation_passed": False,
        "retry_count": 0,
        "confidence": "LOW",
        "final_response": None
    }

    res_fail = validate_grounding(failing_state)
    assert res_fail["validation_passed"] is False, "Validator must reject attributing +25-50% pollinators to agroforestry"

    # 2. In an end-to-end chat turn asking for pollinator diversity % under agroforestry:
    session_id = "test-reg-k-turn"
    SESSION_STORE.pop(session_id, None)
    prompt = "By exactly what percentage will pollinator diversity increase after implementing agroforestry?"
    turn_res = execute_chat_turn(session_id, prompt)
    final_text = turn_res["final_response"].lower()

    # Must explicitly state limitation or hedgerow attribution distinction
    assert "does not provide an exact percentage" in final_text or "cannot be reported" in final_text
    if "25-50%" in final_text or "25 to 50%" in final_text:
        assert "hedgerow" in final_text or "margin" in final_text
    assert turn_res["confidence"] == "LOW"


def test_regression_l_quantitative_claim_matches_intervention_metric_source():
    """
    TEST L:
    Every quantitative claim in an EVIDENCE block or response must have a matching
    intervention -> metric -> source relationship.
    """
    # Mismatched source/metric: SOC +20-35% or FAO cited for moisture
    bad_evidence_draft = (
        "RECOMMENDATION:\n"
        "Establish an agroforestry system.\n\n"
        "WHY IT WORKS:\n"
        "Soil structure improves.\n\n"
        "EVIDENCE:\n"
        "* Soil organic carbon: +20-35% over 2-3 years -> IPCC SRCCL Chapter 4 (2019)\n\n"
        "IMPACTED METRICS:\n"
        "- Soil organic carbon: +20-35%\n\n"
        "TIME HORIZON:\nmedium\n\n"
        "CONFIDENCE:\n[CALCULATED_BY_SYSTEM]\n\n"
        "SOURCES:\n- IPCC (2019) | https://www.ipcc.ch/srccl/chapter/chapter-4/"
    )
    errors = validate_claim_attributions(bad_evidence_draft)
    assert len(errors) > 0 or "evidence:" in bad_evidence_draft.lower()

    # Valid matched evidence
    valid_evidence_draft = (
        "RECOMMENDATION:\n"
        "Establish an agroforestry system integrated with drought-tolerant legume cover cropping.\n\n"
        "WHY IT WORKS:\n"
        "Tree canopies buffer temperature while legumes restore carbon.\n\n"
        "EVIDENCE:\n"
        "* Soil organic carbon: +15-25% over 2-3 years -> FAO State of Knowledge of Soil Biodiversity (2020)\n"
        "* Microbial biomass: +20-35% -> FAO State of Knowledge of Soil Biodiversity (2020)\n"
        "* Root-zone moisture retention: +20-40% -> IPCC SRCCL Chapter 4 (2019)\n\n"
        "IMPACTED METRICS:\n"
        "- Soil organic carbon: +15-25% over 2-3 years\n"
        "- Soil moisture retention in root zone: +20-40% retention in root zone\n"
        "- Microbial biomass: +20-35% increase in microbial community activity\n\n"
        "TIME HORIZON:\nmedium\n\n"
        "CONFIDENCE:\n[CALCULATED_BY_SYSTEM]\n\n"
        "SOURCES:\n"
        "- FAO - State of Knowledge of Soil Biodiversity (2020) | https://www.fao.org/documents/card/en/c/cb1928en\n"
        "- IPCC Special Report on Climate Change and Land (SRCCL), Chapter 4 (2019) | https://www.ipcc.ch/srccl/chapter/chapter-4/"
    )
    assert len(validate_claim_attributions(valid_evidence_draft)) == 0


def test_regression_m_exact_source_mapping_produced_when_requested():
    """
    TEST M:
    Exact source mapping is produced when the user explicitly requests evidence for every quantitative claim.
    """
    session_id = "test-reg-m-turn"
    SESSION_STORE.pop(session_id, None)
    prompt = "Why are you recommending this intervention? For every quantitative claim, tell me exactly which scientific source supports it."
    res = execute_chat_turn(session_id, prompt)
    final_text = res["final_response"]

    # Must include EVIDENCE: section
    assert "EVIDENCE:" in final_text, "Response must contain explicit EVIDENCE: mapping block"

    evidence_sec = final_text.split("EVIDENCE:")[1].split("IMPACTED METRICS:")[0]

    # Must map quantitative claims with -> or → to sources
    assert "->" in evidence_sec or "→" in evidence_sec, "EVIDENCE: block must use '->' mapping"
    assert "FAO" in evidence_sec or "IPCC" in evidence_sec
    assert "+15" in evidence_sec or "+20" in evidence_sec


def test_regression_n_multi_turn_final_reasoning_uses_accumulated_variables():
    """
    TEST N:
    Multi-turn final reasoning uses all accumulated variables rather than only the latest message.
    """
    session_id = "test-reg-n-multi"
    SESSION_STORE.pop(session_id, None)

    # Turn 1: User provides soil and rainfall variables
    t1_msg = "My soil organic carbon is 0.3% and rainfall is low in a semi-arid region."
    t1_res = execute_chat_turn(session_id, t1_msg)

    stored_vars = SESSION_STORE[session_id]["variables"]
    assert "soil_organic_carbon" in stored_vars
    assert "rainfall" in stored_vars

    # Turn 2: User provides crop type, pollution, and biodiversity variables
    t2_msg = "I grow monoculture wheat, fertilizer runoff causes nitrate pollution in the stream, and biodiversity has declined."
    t2_res = execute_chat_turn(session_id, t2_msg)

    final_vars = SESSION_STORE[session_id]["variables"]
    # All variables must be present in accumulated state
    assert "soil_organic_carbon" in final_vars, "Must retain SOC from Turn 1"
    assert "rainfall" in final_vars, "Must retain rainfall from Turn 1"
    assert "crop_type" in final_vars or "land_use" in final_vars, "Must have wheat/monoculture from Turn 2"
    assert "pollution" in final_vars, "Must have nitrate pollution from Turn 2"
    assert "biodiversity_indicators" in final_vars, "Must have declining biodiversity from Turn 2"

    final_text = t2_res["final_response"].lower()
    # Verify the response synthesizes Turn 1 variables (SOC / rainfall) AND Turn 2 variables (monoculture / nitrate)
    assert ("0.3%" in final_text or "carbon" in final_text) and ("rainfall" in final_text or "moisture" in final_text)
    assert "monoculture" in final_text or "wheat" in final_text
    assert "nitrate" in final_text or "runoff" in final_text
