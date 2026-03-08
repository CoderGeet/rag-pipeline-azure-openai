"""
chunker.py — Splits Documents into smaller Chunks for embedding.

Why chunk? LLMs have token limits and embeddings work best on focused
passages. Think of each chunk like a paragraph in a book — small enough
to be specific, big enough to have context.
"""

import uuid
from core.models import Document, Chunk

class TextChunker:
    """
    Splits text using a sliding window approach.

    Parameters:
    ----------
    chunk_size   : target character length per chunk
    chunk_overlap: characters shared between adjacent chunks
                   (helps avoid cutting sentences mid-thought)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int=50):
        if (chunk_overlap >= chunk_size):
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        """Split a single Document into a list of Chunks."""
        text = document.content.strip()
        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break on a sentence boundary
            if end < len(text):
                boundary = text.rfind(".", start, end)
                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    content=chunk_text,
                    index=index,
                    metadata={**document.metadata, "source": document.source},
                ))
                index += 1

            start = end - self.chunk_overlap

        return chunks

    def split_many(self, documents: list[Document]) -> list[Chunk]:
        """Convenience method: split multiple documents at once."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split(doc))
        return all_chunks