"""
Completeness Checking Node.

Determines whether the user has provided sufficient environmental context to generate
an evidence-backed multi-metric recommendation, or whether a focused clarifying
question should be asked.
"""

from typing import Dict, Any
from app.graph.state import GraphState


def check_completeness(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates context completeness based on the query and active variables.
    Returns updated is_complete flag and optional clarifying_question.
    """
    message = state.get("current_user_message", "").lower()
    variables = state.get("variables", {})

    # 1. Pure informational, exact-metric, or evidence verification queries do not require land-specific variables
    informational_phrases = [
        "what is", "how does", "explain", "why do", "why are you", "why is",
        "tell me about", "which source", "what evidence", "can you explain"
    ]
    is_info_prefix = any(message.startswith(phrase) for phrase in informational_phrases)
    is_exact_metric = (
        "by exactly what percentage" in message
        or "what percentage" in message
        or "exact percentage" in message
    )
    is_evidence_query = (
        "for every quantitative claim" in message
        or "which scientific source" in message
        or "tell me exactly which" in message
        or "source supports" in message
        or "evidence:" in message
    )

    # If this is a follow-up in an existing session where an assistant response already exists:
    messages = state.get("messages", [])
    has_prior_assistant_response = any(m.get("role") == "assistant" for m in messages)

    if is_info_prefix or is_exact_metric or is_evidence_query or (has_prior_assistant_response and "why" in message):
        return {
            "is_complete": True,
            "clarifying_question": None
        }

    # 2. Check presence of core environmental variables
    has_soil = bool(variables.get("soil_organic_carbon") or variables.get("soil_ph") or variables.get("soil_moisture"))
    has_water = bool(variables.get("rainfall"))
    has_land = bool(variables.get("land_use") or variables.get("crop_type"))
    has_region = bool(variables.get("region"))
    has_human = bool(variables.get("pollution") or variables.get("deforestation"))

    # Total contextual signals
    context_count = sum([has_soil, has_water, has_land, has_region, has_human])

    # If the user is reporting a problem or seeking recommendations for their land
    # (e.g., "biodiversity is declining", "how can I improve my yield?"):
    if context_count < 2:
        missing_items = []
        if not has_soil:
            missing_items.append("soil organic carbon % (or soil condition)")
        if not has_water:
            missing_items.append("rainfall pattern")
        if not has_land:
            missing_items.append("current land use / crop type")

        clarifying_q = (
            f"To provide a scientifically grounded, multi-metric recommendation for your land, "
            f"could you share your {', '.join(missing_items)}?"
        )
        return {
            "is_complete": False,
            "clarifying_question": clarifying_q
        }

    # Sufficient context present
    return {
        "is_complete": True,
        "clarifying_question": None
    }
