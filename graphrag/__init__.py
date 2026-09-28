"""Minimal GraphRAG: build a knowledge graph from documents and query it.

Pipeline: extract entities/relations -> NetworkX graph -> graph retrieval ->
answer generation. An LLM (Gemini) is used when GOOGLE_API_KEY is set;
otherwise a deterministic dry-run mode keeps everything working offline.
"""

__version__ = "0.1.0"
