import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.workflow import execute_chat_turn
from app.retrieval.retriever import DualLayerRetriever

print("=" * 70)
print("TEST 1: EXACT RETRIEVED EVIDENCE SUPPORTING NUMERICAL CLAIMS")
print("=" * 70)

retrieval = DualLayerRetriever.retrieve(
    query="soil organic carbon = 0.3% rainfall = low crop = monoculture wheat region = semi-arid High synthetic fertilizer runoff is causing nitrate pollution",
    variable_tags=["soil", "climate", "land_use", "human_impact"],
    metric_keys=["soil_organic_carbon", "microbial_biomass", "soil_moisture_retention", "agricultural_runoff_nitrates"]
)

print("\n[A] RETRIEVED metric_facts ROWS:")
for i, fact in enumerate(retrieval["metric_facts"], 1):
    print(f"\nRow {i}:")
    print(f"  - Intervention:   {fact.get('intervention')}")
    print(f"  - Affects Metric: {fact.get('affects_metric')}")
    print(f"  - Effect Value:   {fact.get('effect_value')}")
    print(f"  - Time Horizon:   {fact.get('time_horizon')}")
    print(f"  - Source:         {fact.get('source')}")
    print(f"  - Source URL:     {fact.get('source_url')}")

print("\n[B] RETRIEVED knowledge_chunks ROWS:")
for i, chunk in enumerate(retrieval["knowledge_chunks"], 1):
    print(f"\nChunk {i}:")
    print(f"  - Variable Tag: {chunk.get('variable_tag')}")
    print(f"  - Source:       {chunk.get('source')}")
    print(f"  - Source URL:   {chunk.get('source_url')}")
    print(f"  - Content:      {chunk.get('content')}")

print("\n" + "=" * 70)
print("TEST 2: CONFIDENCE SCORING AUDIT")
print("=" * 70)
print("Auditing matching rules for: 'agroforestry' vs 'legume-based cover crops'...")
for fact in retrieval["metric_facts"]:
    print(f"Fact in DB: Intervention='{fact.get('intervention')}', Metric='{fact.get('affects_metric')}'")

print("\n" + "=" * 70)
print("TEST 3: HALLUCINATION TEST")
print("Prompt: 'My farm has 0.3% soil organic carbon, low rainfall, monoculture wheat, and severe nitrate pollution. By exactly what percentage will biodiversity increase if I implement agroforestry?'")
print("=" * 70)

question_3 = (
    "My farm has 0.3% soil organic carbon, low rainfall, monoculture wheat, and severe nitrate pollution. "
    "By exactly what percentage will biodiversity increase if I implement agroforestry?"
)
turn_3 = execute_chat_turn("fresh-diagnostic-session-3", question_3)

print("\n--- FINAL RESPONSE FOR TEST 3 ---")
print(turn_3.get("final_response"))
print("\n--- METADATA ---")
print("Confidence level:", turn_3.get("confidence"))
print("Validation passed:", turn_3.get("validation_passed"))
print("Retry count:", turn_3.get("retry_count"))
print("Draft response before validation:", turn_3.get("draft_response"))
