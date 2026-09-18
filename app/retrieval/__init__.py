"""
Retrieval package providing vector similarity and structured fact lookup.
"""
from app.retrieval.vector_search import search_knowledge_chunks
from app.retrieval.structured_search import search_metric_facts
from app.retrieval.retriever import DualLayerRetriever

__all__ = ["search_knowledge_chunks", "search_metric_facts", "DualLayerRetriever"]
