"""
Dual-Layer Retrieval Coordinator.

Combines pgvector similarity search on qualitative knowledge_chunks with
structured SQL lookup on quantitative metric_facts, keeping them strictly separated
and preserving variable-by-variable evidence mapping.
"""

from typing import List, Dict, Any, Optional
from app.retrieval.vector_search import search_knowledge_chunks
from app.retrieval.structured_search import search_metric_facts


VARIABLE_DOMAIN_CONFIG = {
    "soil_organic_carbon": {
        "tag": "soil",
        "label": "Soil Health (SOC)",
        "query": "soil health organic carbon microbial biomass legume cover crops",
        "metrics": ["soil_organic_carbon", "microbial_biomass"]
    },
    "soil_ph": {
        "tag": "soil",
        "label": "Soil Health (pH)",
        "query": "soil pH nutrient bioavailability liming organic matter",
        "metrics": ["soil_organic_carbon"]
    },
    "soil_moisture": {
        "tag": "soil",
        "label": "Soil Moisture",
        "query": "soil moisture conservation residue mulching organic matter",
        "metrics": ["soil_moisture_retention"]
    },
    "rainfall": {
        "tag": "climate",
        "label": "Climate & Hydrology (Rainfall)",
        "query": "semi-arid low rainfall drought water retention agroforestry",
        "metrics": ["soil_moisture_retention", "water_holding_capacity", "yield_variability_under_drought"]
    },
    "region": {
        "tag": "climate",
        "label": "Region & Climate Zone",
        "query": "semi-arid dryland climate agroforestry resilience",
        "metrics": ["soil_moisture_retention", "yield_variability_under_drought"]
    },
    "land_use": {
        "tag": "land_use",
        "label": "Land Use & Crop Diversity",
        "query": "monoculture crop rotation agroforestry structural diversity",
        "metrics": ["yield_variability_under_drought", "carbon_sequestration"]
    },
    "crop_type": {
        "tag": "land_use",
        "label": "Crop Type",
        "query": "wheat crop rotation legume intercropping diversified",
        "metrics": ["yield_variability_under_drought", "soil_organic_carbon"]
    },
    "pollution": {
        "tag": "human_impact",
        "label": "Human Impact & Pollution",
        "query": "synthetic fertilizer runoff nitrate pollution riparian buffer strip",
        "metrics": ["agricultural_runoff_nitrates"]
    },
    "biodiversity_indicators": {
        "tag": "biodiversity",
        "label": "Biodiversity Status",
        "query": "biodiversity decline wild pollinator richness microbial biomass hedgerows",
        "metrics": ["pollinator_species_richness", "microbial_biomass"]
    },
    "deforestation": {
        "tag": "human_impact",
        "label": "Deforestation & Land Clearing",
        "query": "deforestation ecological corridors agroforestry habitat fragmentation",
        "metrics": ["carbon_sequestration"]
    }
}


class DualLayerRetriever:
    """
    Coordinates semantic vector retrieval and structured quantitative fact lookups
    while preserving which environmental variable each retrieved item supports.
    """

    @staticmethod
    def retrieve(
        query: str,
        variable_tags: Optional[List[str]] = None,
        metric_keys: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Base dual-layer retrieval method.
        """
        top_k = max(len(variable_tags), 4) if variable_tags else 4
        knowledge_chunks = search_knowledge_chunks(
            query_text=query,
            variable_tags=variable_tags,
            top_k=top_k
        )

        fact_limit = max(len(metric_keys), 6) if metric_keys else 6
        metric_facts = search_metric_facts(
            metrics=metric_keys,
            keywords=keywords,
            limit=fact_limit
        )

        scientific_context_lines = []
        for i, chunk in enumerate(knowledge_chunks, 1):
            scientific_context_lines.append(
                f"[{i}] ({chunk.get('variable_tag', 'general')}) {chunk['content']} "
                f"(Source: {chunk['source']} | URL: {chunk['source_url']})"
            )
        general_scientific_context_str = "\n\n".join(scientific_context_lines) if scientific_context_lines else "No direct scientific context found."

        quantitative_facts_lines = []
        for i, fact in enumerate(metric_facts, 1):
            quantitative_facts_lines.append(
                f"- Intervention: {fact['intervention']}\n"
                f"  Affects: {fact['affects_metric']} -> {fact['effect_value']} ({fact['time_horizon']} term)\n"
                f"  Evidence: {fact['source']} | {fact['source_url']}"
            )
        quantitative_facts_str = "\n".join(quantitative_facts_lines) if quantitative_facts_lines else "No specific quantitative facts found."

        return {
            "knowledge_chunks": knowledge_chunks,
            "metric_facts": metric_facts,
            "general_scientific_context_str": general_scientific_context_str,
            "quantitative_facts_str": quantitative_facts_str,
            "combined_context_prompt": (
                "GENERAL SCIENTIFIC CONTEXT:\n"
                f"{general_scientific_context_str}\n\n"
                "QUANTITATIVE FACTS:\n"
                f"{quantitative_facts_str}"
            )
        }

    @classmethod
    def retrieve_by_variables(
        cls,
        variables: Dict[str, Any],
        user_message: str
    ) -> Dict[str, Any]:
        """
        Performs fine-grained retrieval for each active environmental variable,
        preserving variable-to-evidence mapping for the reasoning engine.
        """
        variable_sections = []
        all_chunks = []
        all_facts = []
        seen_chunk_contents = set()
        seen_fact_keys = set()

        # Iterate over all user-supplied variables
        for var_name, var_value in variables.items():
            if not var_value or var_name not in VARIABLE_DOMAIN_CONFIG:
                continue

            config = VARIABLE_DOMAIN_CONFIG[var_name]
            tag = config["tag"]
            label = config["label"]
            metrics = config["metrics"]
            search_query = f"{config['query']} {var_value}"

            # Retrieve chunks for this specific variable
            chunks = search_knowledge_chunks(query_text=search_query, variable_tags=[tag], top_k=2)
            # Retrieve facts for this specific variable
            facts = search_metric_facts(metrics=metrics, limit=3)

            # Build variable section
            sec_lines = [f"VARIABLE: {label} (User Value: '{var_value}')"]
            sec_lines.append("  Qualitative Scientific Context:")
            for c in chunks:
                sec_lines.append(f"    - {c['content']} [{c['source']}]")
                if c["content"] not in seen_chunk_contents:
                    seen_chunk_contents.add(c["content"])
                    all_chunks.append(c)

            sec_lines.append("  Direct Quantitative Metric Evidence:")
            if facts:
                for f in facts:
                    sec_lines.append(
                        f"    - {f['intervention']} -> {f['affects_metric']}: {f['effect_value']} ({f['time_horizon']} term) [{f['source']}]"
                    )
                    fact_key = (f["intervention"], f["affects_metric"])
                    if fact_key not in seen_fact_keys:
                        seen_fact_keys.add(fact_key)
                        all_facts.append(f)
            else:
                sec_lines.append("    - (No direct quantitative metric_fact row found for this specific variable)")

            variable_sections.append("\n".join(sec_lines))

        # Also perform standard baseline retrieval to guarantee full coverage
        base_tags = list({cfg["tag"] for k, cfg in VARIABLE_DOMAIN_CONFIG.items() if k in variables})
        base_metrics = []
        for k, cfg in VARIABLE_DOMAIN_CONFIG.items():
            if k in variables:
                base_metrics.extend(cfg["metrics"])

        base_res = cls.retrieve(
            query=f"{user_message} {' '.join(str(v) for v in variables.values())}",
            variable_tags=base_tags if base_tags else None,
            metric_keys=base_metrics if base_metrics else None
        )

        # Merge deduplicated items
        for c in base_res["knowledge_chunks"]:
            if c["content"] not in seen_chunk_contents:
                seen_chunk_contents.add(c["content"])
                all_chunks.append(c)

        for f in base_res["metric_facts"]:
            fact_key = (f["intervention"], f["affects_metric"])
            if fact_key not in seen_fact_keys:
                seen_fact_keys.add(fact_key)
                all_facts.append(f)

        variable_breakdown_str = "\n\n".join(variable_sections) if variable_sections else "No variable-specific sections mapped."

        # Format complete evidence block
        combined_prompt = (
            "VARIABLE-BY-VARIABLE SCIENTIFIC EVIDENCE:\n"
            f"{variable_breakdown_str}\n\n"
            "GENERAL SCIENTIFIC CONTEXT (ALL RETRIEVED CHUNKS):\n"
            f"{base_res['general_scientific_context_str']}\n\n"
            "QUANTITATIVE FACTS (ALL RETRIEVED METRIC FACTS):\n"
            f"{base_res['quantitative_facts_str']}"
        )

        return {
            "knowledge_chunks": all_chunks,
            "metric_facts": all_facts,
            "variable_breakdown_str": variable_breakdown_str,
            "general_scientific_context_str": base_res["general_scientific_context_str"],
            "quantitative_facts_str": base_res["quantitative_facts_str"],
            "combined_context_prompt": combined_prompt
        }
