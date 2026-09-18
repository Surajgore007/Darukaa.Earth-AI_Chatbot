# Darukaa.Earth — AI Biodiversity Intelligence Chatbot (Lean MVP)

[![CI/CD Pipeline](https://github.com/Surajgore007/Darukaa.Earth-AI_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Surajgore007/Darukaa.Earth-AI_Chatbot/actions)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)
![PostgreSQL](https://img.shields.io/badge/Supabase-pgvector-3ecf8e.svg)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5--flash-4285F4.svg)

A production-grade, explainable, and scientifically grounded conversational intelligence system that acts as an **AI Environmental Scientist**. Built for the Darukaa.Earth Hackathon / Internship Challenge, this MVP reasons across multiple environmental variables simultaneously (soil, climate, land use, biodiversity, human impact), grounds every recommendation in verifiable scientific evidence (FAO, IPCC, IPBES), asks focused clarifying questions when input is incomplete, and programmatically computes confidence.

---

## 1. Project Overview

Generic LLM chatbots frequently produce shallow, isolated advice (e.g., *"use sustainable practices"* or *"apply compost"*) and invent non-existent statistics. 

The **Darukaa.Earth Chatbot** overcomes this by enforcing:
1. **Multi-Metric Synthesis**: Connecting multiple environmental variables (e.g. Low Soil Organic Carbon + Monoculture Cropping + Low Rainfall + Nitrate Runoff) into **ONE primary integrated intervention**.
2. **Dual-Layer Evidence Grounding**: Strictly separating qualitative scientific mechanisms (`knowledge_chunks`) from quantitative effect sizes (`metric_facts`).
3. **Strict Grounding Validation**: Requiring complete numeric ranges and authentic source citations to exist in retrieved evidence before returning an answer. Rejects unverified numbers with automated retry and conservative fallbacks.
4. **Programmatic Confidence Scoring**: Verifying explicit `(intervention -> metric)` relationship pairs rather than allowing the LLM to self-evaluate.

---

## 2. Challenge Objective

Build an AI-powered conversational system that:
* Maintains a structured, retrievable knowledge base of environmental metrics.
* Understands user queries regarding soil health, climate, land use, biodiversity, and human impact.
* Generates actionable, non-obvious recommendations to restore biodiversity and soil health.
* Connects at least 3 environmental variables together in its reasoning chain.
* Grounds all claims in credible international scientific literature (FAO, IPCC, IPBES).

---

## 3. System Architecture

```text
                             USER QUERY
                                 │
                                 ▼
                         FastAPI Web API
                                 │
                                 ▼
                     LangGraph State Machine
                                 │
                                 ▼
                     extract_variables (Turn Memory)
                                 │
                                 ▼
                     check_completeness
                    ╱                  ╲
        [Context Incomplete]      [Context Sufficient]
                  │                            │
                  ▼                            ▼
        ask_clarification             retrieve_knowledge
     (Single focused question)                 │
                  │              ┌─────────────┴─────────────┐
                  │              ▼                           ▼
                  │      Vector Retrieval            Structured SQL
                  │      (pgvector search)           (metric_facts)
                  │      qualitative chunks        quantitative facts
                  │              └─────────────┬─────────────┘
                  │                            │
                  │                            ▼
                  │                    Gemini Reasoning
                  │               (Step A: Variable implications)
                  │               (Step B: Single primary synthesis)
                  │               (Step C: Strict evidence grounding)
                  │                            │
                  │                            ▼
                  │                   validate_grounding
                  │                  ╱                  ╲
                  │           [Passes Checks]      [Unverified Numbers/URLs]
                  │                  │                      │
                  │                  │            (Retry once with strict prompt)
                  │                  │                      │
                  │                  │             [Fails twice -> Fallback]
                  │                  ▼                      │
                  │           compute_confidence ◄──────────┘
                  │          (HIGH / MEDIUM / LOW)
                  │                  │
                  ▼                  ▼
                  └──────────► FINAL RESPONSE
                                     │
                                     ▼
                                   USER
```

---

## 4. Technology Stack

* **Language**: Python 3.11
* **Web Framework**: FastAPI & Uvicorn
* **Workflow & State**: LangGraph & LangChain Core
* **LLM Engine**: Google Gemini API (`gemini-2.5-flash` for reasoning, `gemini-embedding-001` for embeddings)
* **Database**: Supabase PostgreSQL with `pgvector` extension
* **Database Driver**: `psycopg 3` (binary) + `pgvector-python`
* **Configuration**: `pydantic-settings` & `python-dotenv`
* **Testing**: `pytest`, `pytest-asyncio`, `httpx`
* **User Interface**: Minimalist Vanilla HTML/CSS/JavaScript

---

## 5. Database Design

Hosted on Supabase with the `vector` extension enabled ([app/db/schema.sql](app/db/schema.sql)):

### Table 1: `knowledge_chunks`
Stores peer-reviewed qualitative scientific context for semantic vector search.
* `id`: UUID (Primary Key, default `gen_random_uuid()`)
* `content`: TEXT (Ecology principles, mechanisms, and findings)
* `embedding`: VECTOR(768) (Generated via `gemini-embedding-001`)
* `source`: TEXT (e.g., *"FAO - State of Knowledge of Soil Biodiversity (2020)"*)
* `source_url`: TEXT (Official publication URL)
* `variable_tag`: TEXT (Check constraint: `'soil'`, `'land_use'`, `'biodiversity'`, `'climate'`, `'human_impact'`)
* *Indexes*: IVFFlat vector cosine index (`vector_cosine_ops`) and B-Tree index on `variable_tag`.

### Table 2: `metric_facts`
Stores verified quantitative evidence separately from unstructured text so numeric claims can be checked independently.
* `id`: UUID (Primary Key)
* `intervention`: TEXT (e.g., *"legume-based cover crops"*)
* `affects_metric`: TEXT (e.g., *"soil_organic_carbon"*)
* `effect_value`: TEXT (e.g., *"+15–25% over 2–3 years"*)
* `time_horizon`: TEXT (`'short'`, `'medium'`, or `'long'`)
* `source`: TEXT (e.g., *"FAO (2020)"*)
* `source_url`: TEXT (Official publication URL)
* *Indexes*: B-Tree indexes on `intervention` and `affects_metric`.

---

## 6. Knowledge-Base Design

Curated in [data/scientific_knowledge.py](data/scientific_knowledge.py). Adheres strictly to the **zero fabrication** principle:
* **Soil Health**: Soil organic carbon dynamics, pH nutrient bioavailability, and moisture retention from FAO Soils Portal and World Soil Charter.
* **Land Use / Land Cover**: Monoculture impacts, diversified crop rotations, and reverse phenology agroforestry from IPCC SRCCL Chapter 5 and FAO TECA.
* **Biodiversity Indicators**: Wild pollinator richness, flowering field margins, and soil microbial biomass from IPBES Global Assessment and *Nature Sustainability*.
* **Climate Factors**: Low-rainfall adaptation, microclimate buffering, and drought yield stability from IPCC SRCCL Chapters 2 and 4.
* **Human Impact**: Nitrate/fertilizer runoff filtration and deforestation buffers from *Agriculture, Ecosystems & Environment* and FAO SOFO.

---

## 7. Dual Retrieval Architecture

Vector retrieval and structured SQL lookup remain strictly separated across code and prompt structure:
1. **Vector Search (`search_knowledge_chunks`)**: Queries `knowledge_chunks` using pgvector cosine distance (`<=>`), filtered by user variable tags.
2. **Structured Search (`search_metric_facts`)**: Queries `metric_facts` using parameterized SQL filters on active metrics and interventions.
3. **Distinct Context Streams**: The coordinator (`DualLayerRetriever`) formats results into two distinct prompt blocks:
   * `GENERAL SCIENTIFIC CONTEXT:` (Qualitative ecological principles)
   * `QUANTITATIVE FACTS:` (Exact numerical intervention $\rightarrow$ metric $\rightarrow$ effect values)

---

## 8. LangGraph Conversation Flow

Conversation state is managed as a directed graph in [app/graph/workflow.py](app/graph/workflow.py):
* `extract_variables`: Inspects the latest user message with pattern extractors and updates `variables` without overwriting historical turns.
* `check_completeness`: Analyzes whether sufficient environmental context is present:
  * If insufficient (e.g., only broad problem stated), routes to `ask_clarification` and returns a single focused question.
  * If sufficient, routes to `retrieve_knowledge` $\rightarrow$ `reason` $\rightarrow$ `validate_grounding` $\rightarrow$ `compute_confidence`.

---

## 9. Multi-Variable Reasoning Flow

The reasoning prompt in [app/graph/nodes/reason.py](app/graph/nodes/reason.py) enforces:
1. **Variable Implications**: Analyzes what the variables imply together (e.g. low SOC + low rainfall + monoculture implies degraded microbial biomass and extreme moisture vulnerability).
2. **One Primary Integrated Intervention**: Formulates **ONE** primary integrated strategy (e.g., *"Establish an agroforestry system with integrated legume cover cropping"*), expressly forbidding shopping lists of 5 unrelated practices.
3. **Conservative Language**: Prohibits unestablished exaggerations (e.g. using *"depleted soil organic carbon (0.3%)"* rather than unsupported *"severe soil degradation"*).

---

## 10. Grounding Validator

Implemented in [app/graph/nodes/grounding_validator.py](app/graph/nodes/grounding_validator.py):
* **Complete Expression Matching**: Extracts all percentages, ratios, and timeframes (e.g. `+15–25%`, `20–40%`). Requires the **complete normalized numeric range** to exist in the retrieved evidence. Zero fallback to loose individual digits.
* **Citation Verification**: Ensures all cited URLs match retrieved sources.
* **Retry Mechanism**: On the first violation, Gemini is re-prompted with a strict grounding warning. If the second attempt still contains unverified claims, returns:
  > *"Insufficient evidence for a confident recommendation."*

---

## 11. Programmatic Confidence Scoring

Implemented in [app/graph/nodes/confidence_scorer.py](app/graph/nodes/confidence_scorer.py). Gemini is never permitted to self-rate.
* **HIGH**: The recommended intervention AND the claimed metric match the **same verified row** in `metric_facts`.
* **MEDIUM**: Supported by qualitative `knowledge_chunks`, but lacks an exact matching quantitative metric row.
* **LOW**: Retrieval produced weak support, or the user requested an unverified exact metric.

---

## 12. Example Conversation

### Turn 1 (Incomplete Query)
**User:** *"High synthetic fertilizer runoff is causing nitrate pollution in our local stream."*  
**System:**  
> *"To provide a scientifically grounded, multi-metric recommendation for your land, could you share your soil organic carbon % (or soil condition), rainfall pattern, and current land use / crop type?"*

### Turn 2 (Context Supplied)
**User:** *"Soil organic carbon is 0.3%, rainfall is low, and we grow monoculture wheat in a semi-arid region."*  
**System:**  
```text
RECOMMENDATION:
Establish an agroforestry system with integrated legume cover cropping.

WHY IT WORKS:
Depleted soil organic carbon (0.3%) combined with low rainfall and continuous monoculture wheat limits soil microbial biomass and moisture retention. Introducing nitrogen-fixing trees and legumes creates canopy shade that buffers surface temperatures and increases organic matter. The physical presence of deep tree root systems and cover crop residue improves water infiltration and reduces the reliance on synthetic nitrogen fertilizer, directly curbing agricultural nitrate runoff into local waterways.

IMPACTED METRICS:
- Soil moisture retention in root zone: +20-40% retention in root zone
- Microbial biomass: +20-35% increase in microbial community activity

TIME HORIZON:
medium

CONFIDENCE:
HIGH

SOURCES:
- FAO - State of Knowledge of Soil Biodiversity (2020) | https://www.fao.org/documents/card/en/c/cb1928en
- IPCC Special Report on Climate Change and Land (SRCCL), Chapter 4 (2019) | https://www.ipcc.ch/srccl/chapter/chapter-4/
```

### Exact-Metric Query Handling
**User:** *"By exactly what percentage will biodiversity increase if I implement agroforestry?"*  
**System Output:**  
> *"The retrieved evidence does not provide an exact percentage for biodiversity improvement from agroforestry, so an exact biodiversity percentage cannot be reported. However, retrieved evidence does substantiate related soil moisture retention (+20-40%) and carbon sequestration benefits."*  
> **Confidence:** `LOW`

---

## 13. Scientific Sources Audit

Every piece of quantitative and qualitative evidence is drawn from verifiable sources:
1. **FAO (2020)**: *State of Knowledge of Soil Biodiversity* (`https://www.fao.org/documents/card/en/c/cb1928en`)
2. **FAO (2017)**: *Soil Organic Carbon: the hidden potential* (`https://www.fao.org/documents/card/en/c/bc55b410-67c4-4b47-84f9-b883ea57d9f7`)
3. **FAO (2015)**: *World Soil Charter* (`https://www.fao.org/soils-portal/soil-management/en/`)
4. **FAO TECA**: *Agroforestry and Alley Cropping Guidelines* (`https://www.fao.org/teca/en/technologies/8367`)
5. **FAO (2022)**: *The State of the World's Forests (SOFO)* (`https://www.fao.org/state-of-forests/en/`)
6. **IPCC SRCCL (2019)**: *Special Report on Climate Change and Land* (Chapters 2, 4, 5) (`https://www.ipcc.ch/srccl/`)
7. **IPBES (2019)**: *Global Assessment Report on Biodiversity and Ecosystem Services* (`https://www.ipbes.net/global-assessment`)
8. **Nature Sustainability (2018)**: *Conservation agriculture on soil biodiversity* (`https://www.nature.com/articles/s41893-018-0032-1`)
9. **Agriculture, Ecosystems & Environment (2021)**: *Riparian buffer strips* (`https://doi.org/10.1016/j.agee.2020.107223`)
10. **Global Change Biology (2019)**: *Biochar in semi-arid soils* (`https://doi.org/10.1111/gcb.14583`)

---

## 14. Local Setup

### 1. Prerequisites
* Python 3.11 installed (`py -3.11 --version`).
* Supabase account with PostgreSQL + `pgvector`.

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Surajgore007/Darukaa.Earth-AI_Chatbot.git
cd Darukaa.Earth-AI_Chatbot

# Create virtual environment with Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 15. Environment Variables

Create a `.env` file in the root directory (refer to `.env.example`):
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
DATABASE_URL=postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres
ENVIRONMENT=development
LOG_LEVEL=INFO
```

---

## 16. Running the Application

### 1. Ingest Knowledge into Supabase
```powershell
python scripts/ingest_knowledge.py
```
*(Automatically creates schema, indexes, embeds chunks, and inserts metric facts into Supabase).*

### 2. Start the FastAPI Server
```powershell
python -m uvicorn app.main:app --reload --port 8000
```
* **Web Chat Interface**: `http://localhost:8000`
* **Swagger API Docs**: `http://localhost:8000/docs`

---

## 17. Running Tests

```powershell
# Run full Pytest test suite (33 tests)
pytest -v

# Run 9 challenge benchmark scenarios
python tests/test_queries.py

# Run submission readiness verification scenarios (A through F)
pytest tests/test_submission_scenarios.py -v
```

---

## 18. API Endpoint Documentation

### `GET /health`
Returns system status and model configurations.
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gemini_configured": true,
  "database_configured": true,
  "gemini_model": "gemini-2.5-flash",
  "embedding_model": "gemini-embedding-001"
}
```

### `POST /chat`
Request:
```json
{
  "message": "soil organic carbon = 0.3%, rainfall = low, crop = monoculture wheat, region = semi-arid",
  "session_id": "session-123"
}
```
Response:
```json
{
  "session_id": "session-123",
  "response": "RECOMMENDATION:\nEstablish an agroforestry system with integrated legume cover cropping...\n\nWHY IT WORKS:\nDepleted soil organic carbon (0.3%)...\n\nIMPACTED METRICS:\n- Soil moisture retention: +20-40%\n\nTIME HORIZON:\nmedium\n\nCONFIDENCE:\nHIGH\n\nSOURCES:\n- IPCC SRCCL (2019)...",
  "is_complete": true,
  "confidence": "HIGH",
  "variables": {
    "soil_organic_carbon": "0.3%",
    "rainfall": "low",
    "crop_type": "wheat",
    "region": "semi-arid"
  },
  "clarifying_question": null
}
```

---

## 19. Project Structure

```text
Daruka ai job/
├── .github/workflows/ci.yml         # CI workflow (Pytest + Benchmarks)
├── app/
│   ├── main.py                     # FastAPI application
│   ├── api/routes.py               # /health and /chat endpoints
│   ├── core/config.py              # Pydantic Settings
│   ├── db/client.py & schema.sql   # Supabase pgvector client & schema
│   ├── retrieval/                  # pgvector & structured fact retrieval
│   ├── graph/                      # LangGraph state machine & nodes
│   └── static/index.html           # Lightweight chat UI
├── data/scientific_knowledge.py     # Verified FAO/IPCC datasets
├── scripts/                        # Ingestion & doc generator
└── tests/                          # 33 unit, integration & scenario tests
```

---

## 20. Limitations

1. **Session Volatility**: In-memory session dictionary persists for server process runtime (can be swapped to persistent Supabase session table).
2. **Curated Dataset Size**: Focuses on 11 qualitative chunks and 9 verified quantitative facts across core challenge domains. Expansion can be performed via `scripts/ingest_knowledge.py`.
3. **LLM Provider Dependency**: Tied to Google Gemini (`gemini-2.5-flash`).

---

## 21. Future Improvements

1. **Geospatial Coordinate Lookup**: Ingest GIS raster layers (e.g. SoilGrids, WorldClim) based on user latitude/longitude coordinates.
2. **Persistent Conversation Database**: Store chat session histories in a dedicated Supabase `conversations` table.
3. **Automated Scientific Paper Ingestion**: Implement an automated PDF parser for incoming scientific journal publications.
