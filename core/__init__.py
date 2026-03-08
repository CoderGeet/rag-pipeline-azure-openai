"""
__init__.py — Makes the core folder a Python package.

This allows main.py to write:
    from core import RagPipeline
instead of:
    from core.rag_pipeline import RagPipeline
"""

from core.models import Document, Chunk, EmbeddedChunk, RetrievedChunk, RagResponse
from core.chunker import TextChunker
from core.embedder import AzureEmbedder
from core.vector_store import InMemoryVectorStore
from core.rag_pipeline import RagPipeline

__all__ = [
    "Document", "Chunk", "EmbeddedChunk", "RetrievedChunk", "RagResponse",
    "TextChunker", "AzureEmbedder", "InMemoryVectorStore", "RagPipeline",
]
