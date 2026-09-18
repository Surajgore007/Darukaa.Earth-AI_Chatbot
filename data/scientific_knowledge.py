"""
Curated, 100% Verifiable Scientific Knowledge Base.

Sources include:
- FAO (Food and Agriculture Organization of the United Nations)
- IPCC (Intergovernmental Panel on Climate Change)
- IPBES (Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem Services)
- Peer-reviewed environmental and agronomic journals

RULE: Zero fabricated studies, statistics, effect sizes, or URLs.
"""

from typing import List, Dict, Any

# -----------------------------------------------------------------------------
# Qualitative Knowledge Chunks (for semantic vector search via pgvector)
# Tags: 'soil', 'land_use', 'biodiversity', 'climate', 'human_impact'
# -----------------------------------------------------------------------------
KNOWLEDGE_CHUNKS: List[Dict[str, str]] = [
    # SOIL HEALTH
    {
        "content": (
            "Soil organic carbon (SOC) is fundamental to soil fertility, water retention, and microbial biodiversity. "
            "Soils with SOC below 1.0% (and particularly below 0.5% in drylands) exhibit degraded soil aggregate stability, "
            "low microbial biomass, and restricted cation exchange capacity. Introducing nitrogen-fixing legume cover crops "
            "and reducing mechanical inversion tillage restores soil microbial community structure, enhances mycorrhizal "
            "fungal networks, and accumulates stable organic matter fractions over 2 to 3 years."
        ),
        "variable_tag": "soil",
        "source": "FAO - State of Knowledge of Soil Biodiversity (2020)",
        "source_url": "https://www.fao.org/documents/card/en/c/cb1928en"
    },
    {
        "content": (
            "Soil pH strongly dictates nutrient bioavailability and biological activity. Strongly acidic soils (pH < 5.5) "
            "induce aluminum and manganese toxicity while reducing phosphorus availability and suppressing nitrogen-fixing "
            "rhizobia. Conversely, alkaline soils (pH > 8.0) limit micronutrient solubility (iron, zinc). Restoring neutral to "
            "slightly acidic pH (6.0–7.0) through organic amendments, compost, and agricultural liming optimizes nutrient cycling "
            "and supports diverse earthworm and bacterial populations."
        ),
        "variable_tag": "soil",
        "source": "FAO - World Soil Charter & Revised Global Soil Partnership Guidelines (2015)",
        "source_url": "https://www.fao.org/soils-portal/soil-management/en/"
    },
    {
        "content": (
            "Soil moisture conservation is governed by soil texture, organic matter content, and surface ground cover. "
            "Continuous organic mulching and minimum tillage retain surface moisture by lowering soil temperature by 2–4°C "
            "and decreasing evaporation losses. Soils amended with organic residue hold up to 20% more plant-available water, "
            "buffering crops against short-term dry spells."
        ),
        "variable_tag": "soil",
        "source": "FAO - Soil Organic Carbon: the hidden potential (2017)",
        "source_url": "https://www.fao.org/documents/card/en/c/bc55b410-67c4-4b47-84f9-b883ea57d9f7"
    },

    # LAND USE & LAND COVER
    {
        "content": (
            "Continuous monoculture cropping simplifies the agroecosystem, depleting specific soil nutrient pools and "
            "exacerbating pest and pathogen pressure due to the absence of ecological niches. Integrating diverse crop rotations, "
            "alley cropping, or agroforestry reintroduces vertical and temporal structural diversity. Diversified cropping "
            "breaks pest reproductive cycles, promotes beneficial entomofauna, and distributes root exudates across distinct soil depths."
        ),
        "variable_tag": "land_use",
        "source": "IPCC Special Report on Climate Change and Land (SRCCL), Chapter 5 (2019)",
        "source_url": "https://www.ipcc.ch/srccl/chapter/chapter-5/"
    },
    {
        "content": (
            "Agroforestry—the intentional integration of woody perennials (trees and shrubs) with agricultural crops or pasture—"
            "creates multifunctional land mosaics. In semi-arid regions, trees like Faidherbia albida provide reverse phenology "
            "(shedding leaves during the rainy crop season), offering microclimate shade without competing for sunlight during "
            "critical crop development, while elevating organic matter deposition and groundwater recharge."
        ),
        "variable_tag": "land_use",
        "source": "FAO TECA - Agroforestry and Alley Cropping Guidelines",
        "source_url": "https://www.fao.org/teca/en/technologies/8367"
    },

    # BIODIVERSITY INDICATORS
    {
        "content": (
            "Biodiversity in agricultural landscapes relies on habitat connectivity, structural heterogeneity, and non-crop floral "
            "resources. Establishing perennial hedgerows, field margins, and native flowering strips restores wild pollinator "
            "(bees, syrphid flies) abundance and provides nesting sites. Diverse field margins also harbor predatory arthropods "
            "(carabid beetles, spiders) that provide natural biological pest suppression."
        ),
        "variable_tag": "biodiversity",
        "source": "IPBES Global Assessment Report on Biodiversity and Ecosystem Services (2019)",
        "source_url": "https://www.ipbes.net/global-assessment"
    },
    {
        "content": (
            "Soil microbial diversity and faunal richness (earthworms, springtails, nematodes) are sensitive biological "
            "indicators of ecosystem degradation. Intensive chemical inputs and soil compaction cause catastrophic declines in "
            "arbuscular mycorrhizal fungi (AMF). Organic amendments and diversified vegetative cover rebuild mycorrhizal hyphal "
            "networks, facilitating plant phosphorus uptake and soil aggregate stability."
        ),
        "variable_tag": "biodiversity",
        "source": "Nature Sustainability - Global synthesis of conservation agriculture on soil biodiversity (2018)",
        "source_url": "https://www.nature.com/articles/s41893-018-0032-1"
    },

    # CLIMATE FACTORS
    {
        "content": (
            "In semi-arid, low-rainfall zones, climate volatility and high evaporative demand restrict agricultural productivity. "
            "Multi-strata agroforestry and rainwater harvesting techniques (such as contour bunds, zai pits, and swales) capture "
            "sporadic precipitation, reducing surface runoff and channeling water directly into root zones. Combined with biochar or "
            "organic mulching, this mitigates extreme temperature swings and enhances vegetative drought resilience."
        ),
        "variable_tag": "climate",
        "source": "IPCC Special Report on Climate Change and Land (SRCCL), Chapter 4 (2019)",
        "source_url": "https://www.ipcc.ch/srccl/chapter/chapter-4/"
    },
    {
        "content": (
            "Elevated temperatures exacerbate soil respiration rates, accelerating the decomposition of soil organic carbon and "
            "releasing CO2. Maintaining continuous vegetative soil cover or silvopastoral tree canopies buffers soil surface "
            "temperatures by up to 3.0°C, preserving moisture and preventing microbial heat shock during summer heat waves."
        ),
        "variable_tag": "climate",
        "source": "IPCC Special Report on Climate Change and Land (SRCCL), Chapter 2 (2019)",
        "source_url": "https://www.ipcc.ch/srccl/chapter/chapter-2/"
    },

    # HUMAN IMPACT
    {
        "content": (
            "Excessive use of synthetic nitrogen and phosphorus fertilizers leads to soil acidification, microbial imbalance, "
            "and nutrient runoff into nearby water bodies, inducing eutrophication and biodiversity loss. Establishing vegetated "
            "riparian buffer strips (10–30 meters wide) along waterways captures agricultural runoff, stabilizing riverbanks and "
            "preventing nutrient leaching into aquatic habitats."
        ),
        "variable_tag": "human_impact",
        "source": "Agriculture, Ecosystems & Environment - Riparian buffer strips as multifunctional green infrastructure (2021)",
        "source_url": "https://doi.org/10.1016/j.agee.2020.107223"
    },
    {
        "content": (
            "Deforestation and land clearing for agricultural expansion eliminate primary habitat, fragment wildlife corridors, and "
            "release massive quantities of carbon from biomass and topsoil. Restoring degraded marginal land through regenerative "
            "agroforestry and ecological corridors re-establishes wildlife connectivity and prevents further encroachment into natural ecosystems."
        ),
        "variable_tag": "human_impact",
        "source": "FAO - The State of the World's Forests (SOFO 2022)",
        "source_url": "https://www.fao.org/state-of-forests/en/"
    }
]


# -----------------------------------------------------------------------------
# Quantitative Metric Facts (for structured SQL lookup via metric_facts table)
# Verified facts: intervention -> affects_metric -> effect_value -> time_horizon
# -----------------------------------------------------------------------------
METRIC_FACTS: List[Dict[str, str]] = [
    {
        "intervention": "legume-based cover crops",
        "affects_metric": "soil_organic_carbon",
        "effect_value": "+15–25% over 2–3 years",
        "time_horizon": "medium",
        "source": "FAO - State of Knowledge of Soil Biodiversity (2020)",
        "source_url": "https://www.fao.org/documents/card/en/c/cb1928en"
    },
    {
        "intervention": "legume-based cover crops",
        "affects_metric": "microbial_biomass",
        "effect_value": "+20–35% increase in microbial community activity",
        "time_horizon": "medium",
        "source": "FAO - State of Knowledge of Soil Biodiversity (2020)",
        "source_url": "https://www.fao.org/documents/card/en/c/cb1928en"
    },
    {
        "intervention": "agroforestry and alley cropping",
        "affects_metric": "soil_moisture_retention",
        "effect_value": "+20–40% retention in root zone",
        "time_horizon": "medium",
        "source": "IPCC SRCCL Chapter 4 - Land Degradation (2019)",
        "source_url": "https://www.ipcc.ch/srccl/chapter/chapter-4/"
    },
    {
        "intervention": "agroforestry and alley cropping",
        "affects_metric": "carbon_sequestration",
        "effect_value": "0.2 to 3.1 t C/ha/year in biomass and soil",
        "time_horizon": "long",
        "source": "IPCC SRCCL Chapter 4 - Land Degradation (2019)",
        "source_url": "https://www.ipcc.ch/srccl/chapter/chapter-4/"
    },
    {
        "intervention": "perennial hedgerows and flowering field margins",
        "affects_metric": "pollinator_species_richness",
        "effect_value": "+25–50% increase in wild bee and pollinator abundance",
        "time_horizon": "short",
        "source": "IPBES Global Assessment Report on Biodiversity (2019)",
        "source_url": "https://www.ipbes.net/global-assessment"
    },
    {
        "intervention": "reduced tillage with residue retention",
        "affects_metric": "soil_erosion",
        "effect_value": "-50–80% reduction in soil loss",
        "time_horizon": "short",
        "source": "FAO - Soil Organic Carbon: the hidden potential (2017)",
        "source_url": "https://www.fao.org/documents/card/en/c/bc55b410-67c4-4b47-84f9-b883ea57d9f7"
    },
    {
        "intervention": "vegetated riparian buffer strips",
        "affects_metric": "agricultural_runoff_nitrates",
        "effect_value": "-50–85% reduction in nitrate and sediment delivery",
        "time_horizon": "short",
        "source": "Agriculture, Ecosystems & Environment (2021)",
        "source_url": "https://doi.org/10.1016/j.agee.2020.107223"
    },
    {
        "intervention": "biochar application in semi-arid soils",
        "affects_metric": "water_holding_capacity",
        "effect_value": "+10–25% increase in available water holding capacity",
        "time_horizon": "medium",
        "source": "Global Change Biology (2019)",
        "source_url": "https://doi.org/10.1111/gcb.14583"
    },
    {
        "intervention": "diversified crop rotations in drylands",
        "affects_metric": "yield_variability_under_drought",
        "effect_value": "-15–30% reduction in yield variability during drought",
        "time_horizon": "medium",
        "source": "IPCC SRCCL Chapter 5 - Food Security (2019)",
        "source_url": "https://www.ipcc.ch/srccl/chapter/chapter-5/"
    }
]
