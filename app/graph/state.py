"""
LangGraph state definitions for conversation and environmental tracking.
"""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class EnvironmentalVariables(BaseModel):
    """
    Tracks environmental variables extracted across conversation turns.
    """
    soil_organic_carbon: Optional[str] = Field(None, description="Soil organic carbon percentage or level (e.g., '0.3%')")
    soil_ph: Optional[str] = Field(None, description="Soil pH level (e.g., '5.5', 'alkaline')")
    soil_moisture: Optional[str] = Field(None, description="Soil moisture condition (e.g., 'low', 'dry', 'waterlogged')")
    rainfall: Optional[str] = Field(None, description="Rainfall pattern or amount (e.g., 'low', 'semi-arid', 'seasonal')")
    land_use: Optional[str] = Field(None, description="Land use type (e.g., 'monoculture', 'pasture', 'cropland')")
    crop_type: Optional[str] = Field(None, description="Crop or vegetation type (e.g., 'wheat', 'corn', 'soy')")
    region: Optional[str] = Field(None, description="Geographic region or climate zone (e.g., 'semi-arid', 'temperate')")
    temperature: Optional[str] = Field(None, description="Temperature conditions (e.g., 'high heat', 'frost-prone')")
    biodiversity_indicators: Optional[str] = Field(None, description="Biodiversity status or observations (e.g., 'declining', 'low pollinators')")
    pollution: Optional[str] = Field(None, description="Pollution factors (e.g., 'nitrate runoff', 'heavy fertilizer')")
    deforestation: Optional[str] = Field(None, description="Deforestation or land clearing status")

    def count_present_variables(self) -> int:
        """Count how many variables currently have non-null values."""
        return sum(1 for v in self.model_dump().values() if v is not None)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return non-null variables as a clean dictionary."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class GraphState(TypedDict):
    """
    Unified state dictionary passed through the LangGraph conversation workflow.
    """
    session_id: str
    current_user_message: str
    messages: List[Dict[str, str]]
    variables: Dict[str, Any]
    is_complete: bool
    clarifying_question: Optional[str]
    retrieved_context: Dict[str, Any]
    draft_response: Optional[str]
    validation_passed: bool
    retry_count: int
    confidence: str
    final_response: Optional[str]
