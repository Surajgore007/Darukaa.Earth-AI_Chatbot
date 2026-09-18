"""
Knowledge Retrieval Node for LangGraph.

Prepares query parameters from active environmental variables and invokes
the DualLayerRetriever to obtain separated qualitative chunks and quantitative facts.
"""

from typing import Dict, Any, List
from app.graph.state import GraphState
from app.retrieval.retriever import DualLayerRetriever


def retrieve_knowledge(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves scientific knowledge and quantitative metric facts.
    """
    message = state.get("current_user_message", "")
    variables = state.get("variables", {})

    # Map extracted variables to tags for vector filtering
    variable_tags: List[str] = []
    if any(k in variables for k in ["soil_organic_carbon", "soil_ph", "soil_moisture"]):
        variable_tags.append("soil")
    if any(k in variables for k in ["rainfall", "temperature", "region"]):
        variable_tags.append("climate")
    if any(k in variables for k in ["land_use", "crop_type"]):
        variable_tags.append("land_use")
    if "biodiversity_indicators" in variables:
        variable_tags.append("biodiversity")
    if any(k in variables for k in ["pollution", "deforestation"]):
        variable_tags.append("human_impact")

    # If no specific tags mapped, search broadly
    if not variable_tags:
        variable_tags = ["soil", "land_use", "biodiversity", "climate"]

    # Target relevant metrics based on user context
    metric_keys: List[str] = []
    if "soil_organic_carbon" in variables:
        metric_keys.extend(["soil_organic_carbon", "microbial_biomass"])
    if "rainfall" in variables:
        metric_keys.extend(["soil_moisture_retention", "water_holding_capacity", "yield_variability_under_drought"])
    if "biodiversity_indicators" in variables:
        metric_keys.append("pollinator_species_richness")
    if "pollution" in variables:
        metric_keys.append("agricultural_runoff_nitrates")

    # If this is a source mapping or evidence request, ensure core metrics are targeted
    msg_lower = message.lower()
    if "source" in msg_lower or "claim" in msg_lower or "evidence" in msg_lower:
        for k in ["soil_organic_carbon", "microbial_biomass", "soil_moisture_retention"]:
            if k not in metric_keys:
                metric_keys.append(k)
        for t in ["soil", "climate", "land_use"]:
            if t not in variable_tags:
                variable_tags.append(t)
    if "pollinator" in msg_lower or "bee" in msg_lower:
        if "pollinator_species_richness" not in metric_keys:
            metric_keys.append("pollinator_species_richness")
        if "biodiversity" not in variable_tags:
            variable_tags.append("biodiversity")

    # Keywords for structured lookup
    keywords: List[str] = []
    if variables.get("land_use"):
        keywords.append(variables["land_use"])
    if variables.get("crop_type"):
        keywords.append(variables["crop_type"])

    # Execute dual-layer retrieval
    if variables:
        retrieval_results = DualLayerRetriever.retrieve_by_variables(
            variables=variables,
            user_message=message
        )
    else:
        retrieval_results = DualLayerRetriever.retrieve(
            query=message,
            variable_tags=variable_tags,
            metric_keys=metric_keys if metric_keys else None,
            keywords=keywords if keywords else None
        )

    return {"retrieved_context": retrieval_results}
