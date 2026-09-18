"""
Variable Extraction Node.

Extracts environmental variables from user input using simple, transparent
regex patterns and keyword heuristics. Merges newly found variables into the
conversation state while strictly preserving data from previous turns.
"""

import re
from typing import Dict, Any
from app.graph.state import GraphState


def extract_variables(state: GraphState) -> Dict[str, Any]:
    """
    Parses state['current_user_message'], extracts environmental variables,
    and updates state['variables'] without overwriting previous turns.
    """
    message = state.get("current_user_message", "").strip()
    msg_lower = message.lower()
    
    # Start with existing variables from previous turns
    existing_vars = dict(state.get("variables", {}))
    new_vars = {}

    # 1. Soil Organic Carbon (SOC)
    soc_match = re.search(r"(?:soil\s+organic\s+carbon|soc)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?\s*%)", msg_lower)
    if not soc_match:
        soc_match = re.search(r"([0-9]+(?:\.[0-9]+)?\s*%)\s*(?:soil\s+organic\s+carbon|soc)", msg_lower)
    if soc_match:
        new_vars["soil_organic_carbon"] = soc_match.group(1).strip()
    elif "low carbon" in msg_lower or "depleted carbon" in msg_lower:
        new_vars["soil_organic_carbon"] = "low"

    # 2. Soil pH
    ph_match = re.search(r"(?:soil\s+)?ph\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)", msg_lower)
    if ph_match:
        new_vars["soil_ph"] = ph_match.group(1).strip()
    elif "acidic" in msg_lower:
        new_vars["soil_ph"] = "acidic"
    elif "alkaline" in msg_lower:
        new_vars["soil_ph"] = "alkaline"

    # 3. Soil Moisture
    moisture_match = re.search(r"(?:soil\s+)?moisture\s*(?:is|=|:)?\s*([a-zA-Z]+)", msg_lower)
    if moisture_match and moisture_match.group(1) not in {"and", "the", "in", "is"}:
        new_vars["soil_moisture"] = moisture_match.group(1).strip()
    elif "dry soil" in msg_lower or "low moisture" in msg_lower:
        new_vars["soil_moisture"] = "low"

    # 4. Rainfall
    rain_match = re.search(r"rainfall\s*(?:is|=|:)?\s*([a-zA-Z0-9\s]+?)(?:,|;|\.|\sand\s|$)", msg_lower)
    if rain_match:
        val = rain_match.group(1).strip()
        if val in {"low", "high", "moderate", "scanty", "irregular", "seasonal", "semi-arid"}:
            new_vars["rainfall"] = val
    if not new_vars.get("rainfall"):
        m_rain = re.search(r"rainfall\s*(?:is|=|:)?\s*(low|high|moderate|scanty|irregular|seasonal)", msg_lower)
        if m_rain:
            new_vars["rainfall"] = m_rain.group(1)
        elif "low rainfall" in msg_lower or "scanty rainfall" in msg_lower:
            new_vars["rainfall"] = "low"
        elif "high rainfall" in msg_lower:
            new_vars["rainfall"] = "high"

    # 5. Land Use & Crop Type
    crop_match = re.search(r"crop\s*(?:is|=|:)?\s*([a-zA-Z0-9\s]+?)(?:,|;|\.|\sand\s|$)", msg_lower)
    if crop_match:
        new_vars["crop_type"] = crop_match.group(1).strip()
    
    if "monoculture" in msg_lower:
        new_vars["land_use"] = "monoculture"
        if "monoculture wheat" in msg_lower:
            new_vars["crop_type"] = "wheat"
    elif "pasture" in msg_lower or "grazing" in msg_lower:
        new_vars["land_use"] = "pasture"
    elif "agroforestry" in msg_lower:
        new_vars["land_use"] = "agroforestry"

    # 6. Region
    region_match = re.search(r"region\s*(?:is|=|:)?\s*([a-zA-Z0-9\s-]+?)(?:,|;|\.|\sand\s|$)", msg_lower)
    if region_match:
        new_vars["region"] = region_match.group(1).strip()
    elif "semi-arid" in msg_lower:
        new_vars["region"] = "semi-arid"
    elif "tropical" in msg_lower:
        new_vars["region"] = "tropical"
    elif "arid" in msg_lower:
        new_vars["region"] = "arid"

    # 7. Temperature
    temp_match = re.search(r"temperature\s*(?:is|=|:)?\s*([a-zA-Z0-9\s]+?)(?:,|;|\.|\sand\s|$)", msg_lower)
    if temp_match:
        new_vars["temperature"] = temp_match.group(1).strip()
    elif "high heat" in msg_lower or "extreme heat" in msg_lower:
        new_vars["temperature"] = "high heat"

    # 8. Biodiversity Indicators
    if "pollinator" in msg_lower or "bee" in msg_lower:
        new_vars["biodiversity_indicators"] = "pollinators"
    elif any(p in msg_lower for p in [
        "biodiversity is declining", "declining biodiversity", "biodiversity declining",
        "biodiversity has declined", "biodiversity declined", "biodiversity decline",
        "loss of biodiversity", "biodiversity loss"
    ]):
        new_vars["biodiversity_indicators"] = "declining"

    # 9. Human Impact & Pollution
    if "nitrate" in msg_lower and ("pollution" in msg_lower or "stream" in msg_lower or "runoff" in msg_lower):
        new_vars["pollution"] = "nitrate pollution in stream from synthetic fertilizer runoff"
    elif "pesticide" in msg_lower or "chemical runoff" in msg_lower or "fertilizer runoff" in msg_lower:
        new_vars["pollution"] = "agricultural runoff from synthetic inputs"

    # 10. Deforestation
    if "deforestation" in msg_lower or "clearing trees" in msg_lower or "cleared land" in msg_lower:
        new_vars["deforestation"] = "deforestation / land clearing"

    # Merge: Update existing with new, strictly preserving previous turns
    for k, v in new_vars.items():
        existing_vars[k] = v

    return {"variables": existing_vars}
