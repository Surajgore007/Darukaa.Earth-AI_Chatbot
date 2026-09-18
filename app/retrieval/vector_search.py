"""
Vector similarity search module for knowledge_chunks table using pgvector.
"""

from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.db.client import get_db_connection
from data.scientific_knowledge import KNOWLEDGE_CHUNKS


def search_knowledge_chunks(
    query_text: str,
    variable_tags: Optional[List[str]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant qualitative scientific knowledge chunks using
    cosine similarity via pgvector on Supabase.
    
    If the database is not configured (e.g. during local tests without credentials),
    falls back gracefully to searching the curated in-memory scientific knowledge base.
    """
    # 1. Live Supabase pgvector retrieval if configured
    if settings.is_database_configured and settings.is_gemini_configured:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedder = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=settings.gemini_api_key,
                output_dimensionality=768
            )
            query_embedding = embedder.embed_query(query_text)

            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    if variable_tags:
                        sql = """
                            SELECT id, content, source, source_url, variable_tag,
                                   (1 - (embedding <=> %s::vector)) AS similarity
                            FROM knowledge_chunks
                            WHERE variable_tag = ANY(%s)
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s;
                        """
                        cur.execute(sql, (query_embedding, variable_tags, query_embedding, top_k))
                    else:
                        sql = """
                            SELECT id, content, source, source_url, variable_tag,
                                   (1 - (embedding <=> %s::vector)) AS similarity
                            FROM knowledge_chunks
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s;
                        """
                        cur.execute(sql, (query_embedding, query_embedding, top_k))

                    rows = cur.fetchall()
                    if rows:
                        return [
                            {
                                "id": str(r["id"]),
                                "content": r["content"],
                                "source": r["source"],
                                "source_url": r["source_url"],
                                "variable_tag": r["variable_tag"],
                                "similarity": float(r["similarity"])
                            }
                            for r in rows
                        ]
        except Exception as e:
            # Fall back to curated dataset if remote DB query fails
            print(f"[WARN] Database vector search fallback activated: {e}")

    # 2. In-memory fallback matching
    # Matches chunks by variable tags and relevance score
    results = []
    query_lower = query_text.lower()
    
    for chunk in KNOWLEDGE_CHUNKS:
        tag_match = False
        if variable_tags:
            tag_match = chunk["variable_tag"] in variable_tags
        
        # Calculate a simple lexical overlap score for fallback
        chunk_text_lower = chunk["content"].lower()
        words = [w for w in query_lower.split() if len(w) > 3]
        matches = sum(1 for w in words if w in chunk_text_lower)
        score = (matches / max(len(words), 1)) + (0.5 if tag_match else 0.0)

        results.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "source_url": chunk["source_url"],
            "variable_tag": chunk["variable_tag"],
            "similarity": round(score, 3)
        })

    # Sort by score descending and take top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
