"""
embedder.py — Converts text into embedding vectors via Azure OpenAI.

An embedding is just a list of ~1536 floats that capture the *meaning* of
text. Similar sentences end up close together in that vector space — which
is what lets us do semantic search later.
"""

import os
from openai import AzureOpenAI
from core.models import Chunk, EmbeddedChunk


class AzureEmbedder:
    """
    Wraps the Azure OpenAI Embeddings API.

    Usage
    -----
        embedder = AzureEmbedder()
        embedded = embedder.embed_chunks(chunks)
    """

    def __init__(self):
        self._client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        self._deployment = os.environ.get(
            "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string — used for embedding the user query."""
        response = self._client.embeddings.create(
            input=text,
            model=self._deployment,
        )
        return response.data[0].embedding

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of Chunks in one batch API call.
        More efficient than embedding one chunk at a time.
        """
        if not chunks:
            return []

        texts = [c.content for c in chunks]

        response = self._client.embeddings.create(
            input=texts,
            model=self._deployment,
        )

        return [
            EmbeddedChunk(chunk=chunk, embedding=item.embedding)
            for chunk, item in zip(chunks, response.data)
        ]
