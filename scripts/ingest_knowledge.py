"""
Knowledge Ingestion Script.

Loads curated scientific knowledge and quantitative metric facts into Supabase PostgreSQL.
Uses Google's currently supported embedding model (gemini-embedding-001) for vector embeddings.
"""

import sys
import os
import argparse
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.db.client import get_db_connection, initialize_schema
from data.scientific_knowledge import KNOWLEDGE_CHUNKS, METRIC_FACTS


def get_embedding_client():
    """
    Initializes and returns the LangChain Google GenAI embeddings client.
    Uses currently supported gemini-embedding-001 with 768 output dimensions
    to align with pgvector table constraints and cosine indexes.
    """
    if not settings.is_gemini_configured:
        raise ValueError("GEMINI_API_KEY is not configured in .env.")
    
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=768
    )


def ingest_data(dry_run: bool = False):
    """
    Ingests scientific chunks and quantitative facts into Supabase.
    """
    print("=" * 60)
    print("Darukaa.Earth — Scientific Knowledge Ingestion")
    print("=" * 60)
    print(f"Total qualitative chunks to ingest: {len(KNOWLEDGE_CHUNKS)}")
    print(f"Total quantitative facts to ingest:  {len(METRIC_FACTS)}")

    if dry_run:
        print("\n[DRY RUN] Validating data structures only (no database writes)...")
        for chunk in KNOWLEDGE_CHUNKS:
            assert chunk["variable_tag"] in {"soil", "land_use", "biodiversity", "climate", "human_impact"}
            assert len(chunk["content"]) > 20
            assert chunk["source"] and chunk["source_url"]
        for fact in METRIC_FACTS:
            assert fact["time_horizon"] in {"short", "medium", "long"}
            assert fact["intervention"] and fact["affects_metric"] and fact["effect_value"]
        print("[DRY RUN] All data structures validated successfully!")
        return

    if not settings.is_database_configured:
        print("\n[ERROR] DATABASE_URL is not set in .env.")
        print("Please configure your Supabase connection string in .env before running live ingestion.")
        sys.exit(1)

    # 1. Initialize DB schema
    print("\n1. Initializing database schema and pgvector extension...")
    initialize_schema()
    print("   [OK] Schema initialized successfully.")

    # 2. Generate embeddings & insert qualitative chunks
    print("\n2. Embedding and inserting qualitative knowledge chunks...")
    embedder = get_embedding_client()
    
    texts = [c["content"] for c in KNOWLEDGE_CHUNKS]
    print(f"   Generating embeddings using '{settings.gemini_embedding_model}' (dim: 768)...")
    embeddings = embedder.embed_documents(texts)
    print(f"   [OK] Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])}).")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Clear existing data to avoid duplicates during re-ingestion
            cur.execute("DELETE FROM knowledge_chunks;")
            cur.execute("DELETE FROM metric_facts;")

            # Insert knowledge chunks
            chunk_query = """
                INSERT INTO knowledge_chunks (content, embedding, source, source_url, variable_tag)
                VALUES (%s, %s, %s, %s, %s);
            """
            for chunk, emb in zip(KNOWLEDGE_CHUNKS, embeddings):
                cur.execute(chunk_query, (
                    chunk["content"],
                    emb,
                    chunk["source"],
                    chunk["source_url"],
                    chunk["variable_tag"]
                ))
            print(f"   [OK] Inserted {len(KNOWLEDGE_CHUNKS)} knowledge chunks into 'knowledge_chunks'.")

            # 3. Insert quantitative metric facts
            print("\n3. Inserting quantitative metric facts...")
            fact_query = """
                INSERT INTO metric_facts (intervention, affects_metric, effect_value, time_horizon, source, source_url)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            for fact in METRIC_FACTS:
                cur.execute(fact_query, (
                    fact["intervention"],
                    fact["affects_metric"],
                    fact["effect_value"],
                    fact["time_horizon"],
                    fact["source"],
                    fact["source_url"]
                ))
            print(f"   [OK] Inserted {len(METRIC_FACTS)} metric facts into 'metric_facts'.")

    print("\n==================================================")
    print("[SUCCESS] Knowledge base ingestion complete!")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest verified knowledge into Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without writing to database")
    args = parser.parse_args()
    ingest_data(dry_run=args.dry_run)
