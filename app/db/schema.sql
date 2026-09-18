-- =============================================================================
-- Darukaa.Earth Biodiversity Intelligence Chatbot — Database Schema
-- Hosted PostgreSQL with pgvector (Supabase)
-- =============================================================================

-- Enable the pgvector extension for semantic vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- TABLE 1: knowledge_chunks
-- Stores qualitative scientific context, peer-reviewed explanations, and guidelines.
-- Retrieved via pgvector cosine distance (<=>).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(768),
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    variable_tag TEXT NOT NULL CHECK (
        variable_tag IN ('soil', 'land_use', 'biodiversity', 'climate', 'human_impact')
    ),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast cosine similarity search with pgvector
CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx 
ON knowledge_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

-- Index for filtering by environmental variable tag
CREATE INDEX IF NOT EXISTS knowledge_chunks_variable_tag_idx 
ON knowledge_chunks (variable_tag);


-- -----------------------------------------------------------------------------
-- TABLE 2: metric_facts
-- Stores quantitative, verified scientific facts (intervention -> metric -> effect).
-- Kept separate from knowledge_chunks so numerical claims can be validated independently.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metric_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    intervention TEXT NOT NULL,
    affects_metric TEXT NOT NULL,
    effect_value TEXT NOT NULL,
    time_horizon TEXT NOT NULL CHECK (
        time_horizon IN ('short', 'medium', 'long')
    ),
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast structured lookup on intervention and affected metric
CREATE INDEX IF NOT EXISTS metric_facts_intervention_idx 
ON metric_facts (intervention);

CREATE INDEX IF NOT EXISTS metric_facts_affects_metric_idx 
ON metric_facts (affects_metric);
