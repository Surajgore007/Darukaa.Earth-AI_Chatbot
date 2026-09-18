"""
Unit tests for Phase 2: Scientific Knowledge Data Integrity & Ingestion logic.
"""

import pytest
from data.scientific_knowledge import KNOWLEDGE_CHUNKS, METRIC_FACTS
from scripts.ingest_knowledge import ingest_data


def test_knowledge_chunks_coverage():
    """Verify that knowledge chunks cover all 5 required challenge domains."""
    required_tags = {"soil", "land_use", "biodiversity", "climate", "human_impact"}
    found_tags = {chunk["variable_tag"] for chunk in KNOWLEDGE_CHUNKS}
    
    assert required_tags.issubset(found_tags), f"Missing required tags: {required_tags - found_tags}"
    assert len(KNOWLEDGE_CHUNKS) >= 10, "Knowledge base should have adequate coverage"


def test_knowledge_chunks_sources():
    """Verify all qualitative chunks have verified sources and URLs."""
    for chunk in KNOWLEDGE_CHUNKS:
        assert chunk["source"], "Source must not be empty"
        assert chunk["source_url"].startswith("http"), f"Invalid source URL in chunk: {chunk['source']}"
        assert len(chunk["content"]) > 50, "Chunk content must be substantive"


def test_metric_facts_structure():
    """Verify all metric facts have valid interventions, metrics, values, horizons, and sources."""
    valid_horizons = {"short", "medium", "long"}
    for fact in METRIC_FACTS:
        assert fact["intervention"], "Intervention must not be empty"
        assert fact["affects_metric"], "affects_metric must not be empty"
        assert fact["effect_value"], "effect_value must not be empty"
        assert fact["time_horizon"] in valid_horizons, f"Invalid time horizon: {fact['time_horizon']}"
        assert fact["source"], "Source must not be empty"
        assert fact["source_url"].startswith("http"), f"Invalid URL in metric fact: {fact['source']}"


def test_ingestion_dry_run():
    """Verify ingestion validation passes in dry run mode."""
    # Should complete without error
    ingest_data(dry_run=True)
