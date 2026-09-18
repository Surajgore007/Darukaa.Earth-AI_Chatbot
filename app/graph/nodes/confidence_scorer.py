"""
Confidence Scorer Node.

Programmatically computes the confidence level of the recommendation.
Gemini is NEVER allowed to self-rate.

Scoring Rules:
- HIGH:   Recommendation has a verified (intervention -> metric) relationship supported
          by a matching metric_facts row in the retrieved evidence.
- MEDIUM: Recommendation is supported by relevant qualitative knowledge_chunks, but lacks
          a direct matching quantitative (intervention -> metric) fact.
- LOW:    Retrieval produced only weak support, or the requested metric was unverified/insufficient.
"""

import re
from typing import Dict, Any, List
from app.graph.state import GraphState


def extract_impacted_metrics_from_text(text: str) -> str:
    """
    Extracts the IMPACTED METRICS block from the response.
    """
    match = re.search(r"IMPACTED METRICS:\s*(.*?)(?=\n[A-Z\s]+:|$)", text, re.DOTALL)
    if match:
        return match.group(1).lower()
    return ""


def extract_recommendation_from_text(text: str) -> str:
    """
    Extracts the primary recommendation text.
    """
    match = re.search(r"RECOMMENDATION:\s*(.*?)(?=\n[A-Z\s]+:|$)", text, re.DOTALL)
    if match:
        return match.group(1).lower()
    return text.lower()


def compute_confidence(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates whether an actual (intervention -> metric) relationship from metric_facts
    supports the response, rather than loosely checking isolated keywords.
    """
    final_response = state.get("final_response", "") or state.get("draft_response", "")
    retrieved_data = state.get("retrieved_context", {})

    metric_facts = retrieved_data.get("metric_facts", [])
    knowledge_chunks = retrieved_data.get("knowledge_chunks", [])

    response_lower = final_response.lower()

    # Rule 1: Insufficient evidence or unanswerable query -> LOW
    if (
        "insufficient evidence" in response_lower
        or "cannot be reported" in response_lower
        or "does not provide an exact percentage" in response_lower
    ):
        confidence = "LOW"

    # Rule 2: Verify intervention -> metric pair relationship and variable coverage
    elif len(metric_facts) > 0:
        rec_text = extract_recommendation_from_text(final_response)
        impacted_metrics_text = extract_impacted_metrics_from_text(final_response)

        # Count active user environmental variables
        user_vars = state.get("variables", {})
        active_vars = [k for k, v in user_vars.items() if v]
        num_vars = len(active_vars)

        verified_metrics = set()
        for fact in metric_facts:
            intervention_raw = fact.get("intervention", "").lower()
            metric_raw = fact.get("affects_metric", "").lower().replace("_", " ")

            # Split compound intervention name into core keywords (e.g. 'agroforestry' from 'agroforestry and alley cropping')
            intervention_keywords = [
                k.strip() for k in re.split(r",|\band\b|with", intervention_raw) if len(k.strip()) > 3
            ]
            
            # Check: Does the recommendation contain the intervention?
            intervention_matches = (
                any(k in rec_text for k in intervention_keywords) or
                (intervention_raw in rec_text) or
                (intervention_raw in response_lower)
            )

            # Check: Does the IMPACTED METRICS section (or response) contain this specific metric?
            metric_matches = (
                (metric_raw in impacted_metrics_text) if impacted_metrics_text
                else (metric_raw in response_lower)
            )

            # Both intervention AND affected metric must match the SAME row!
            if intervention_matches and metric_matches:
                verified_metrics.add(fact.get("affects_metric"))

        verified_count = len(verified_metrics)

        # Confidence reflects evidence coverage of the recommendation across variables:
        if num_vars >= 4:
            # Multi-variable problem (4-5 variables): single metric match does NOT earn HIGH!
            if verified_count >= 3:
                confidence = "HIGH"
            elif verified_count in {1, 2}:
                confidence = "MEDIUM"
            elif len(knowledge_chunks) > 0:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        elif num_vars == 3:
            if verified_count >= 2:
                confidence = "HIGH"
            elif verified_count == 1:
                confidence = "MEDIUM"
            elif len(knowledge_chunks) > 0:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            # Focused single/dual-variable query
            if verified_count >= 1:
                confidence = "HIGH"
            elif len(knowledge_chunks) > 0:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

    elif len(knowledge_chunks) > 0:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Inject the computed confidence into the response text replacing placeholder
    if "[CALCULATED_BY_SYSTEM]" in final_response:
        updated_response = final_response.replace("[CALCULATED_BY_SYSTEM]", confidence)
    elif "CONFIDENCE:\n" in final_response:
        updated_response = re.sub(
            r"CONFIDENCE:\s*(\[[^\]]+\]|[a-zA-Z]+)?",
            f"CONFIDENCE:\n{confidence}",
            final_response
        )
    else:
        updated_response = f"{final_response}\n\nCONFIDENCE:\n{confidence}"

    return {
        "confidence": confidence,
        "final_response": updated_response
    }
