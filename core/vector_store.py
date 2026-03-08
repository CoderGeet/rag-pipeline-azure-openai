"""
vector_store.py — In-memory vector store backed by FAISS.

FAISS is like a Dictionary<int, float[]> on steroids — it can search
millions of vectors in milliseconds using nearest-neighbour algorithms.

For production, swap this for Azure AI Search. The interface stays
the same — only this class changes.
"""

import numpy as np
import faiss

from core.models import EmbeddedChunk, Chunk, RetrievedChunk


class InMemoryVectorStore:
    """
    Stores EmbeddedChunks and supports cosine-similarity search.

    Think of it as an IRepository<EmbeddedChunk> with a Search() method.
    """

    def __init__(self):
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []

    @property
    def count(self) -> int:
        return len(self._chunks)

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Index a batch of EmbeddedChunks."""
        if not embedded_chunks:
            return

        vectors = np.array(
            [ec.embedding for ec in embedded_chunks], dtype="float32"
        )

        # Normalise so inner-product == cosine similarity
        faiss.normalize_L2(vectors)

        if self._index is None:
            dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dimension)

        self._index.add(vectors)
        self._chunks.extend(ec.chunk for ec in embedded_chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[RetrievedChunk]:
        """
        Find the top_k most similar chunks to a query vector.
        Returns results sorted by descending similarity score.
        """
        if self._index is None or self.count == 0:
            return []

        query_vec = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(
                chunk=self._chunks[idx],
                score=float(score),
            ))

        return results