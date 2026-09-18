"""
Unit tests for Phase 3: Dual-Layer Retrieval Engine.
"""

import pytest
from app.retrieval.vector_search import search_knowledge_chunks
from app.retrieval.structured_search import search_metric_facts
from app.retrieval.retriever import DualLayerRetriever


def test_vector_search_soil_query():
    """Test qualitative retrieval for soil health queries."""
    results = search_knowledge_chunks(
        query_text="low soil organic carbon and degraded soil biology",
        variable_tags=["soil"],
        top_k=2
    )
    assert len(results) > 0
    # Top result should mention organic carbon or soil biodiversity
    top_content = results[0]["content"].lower()
    assert "carbon" in top_content or "soil" in top_content
    assert results[0]["variable_tag"] == "soil"


def test_structured_search_metrics():
    """Test structured SQL/fact lookup for specific metrics."""
    results = search_metric_facts(
        metrics=["soil_organic_carbon"],
        limit=2
    )
    assert len(results) > 0
    assert results[0]["affects_metric"] == "soil_organic_carbon"
    assert "+15–25%" in results[0]["effect_value"] or "carbon" in results[0]["intervention"].lower()


def test_dual_layer_retriever_separation():
    """Test that qualitative and quantitative contexts are strictly kept separate."""
    res = DualLayerRetriever.retrieve(
        query="semi-arid monoculture wheat low rainfall soil carbon 0.3%",
        variable_tags=["soil", "climate", "land_use"],
        metric_keys=["soil_organic_carbon", "soil_moisture_retention"],
        keywords=["cover crops", "agroforestry"]
    )
    
    combined = res["combined_context_prompt"]
    assert "GENERAL SCIENTIFIC CONTEXT:" in combined
    assert "QUANTITATIVE FACTS:" in combined
    assert len(res["knowledge_chunks"]) > 0
    assert len(res["metric_facts"]) > 0
