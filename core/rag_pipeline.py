"""
rag_pipeline.py — Orchestrates all RAG components end-to-end.

This is the equivalent of a Service class in a .NET application.
It depends on the other core classes and wires them together —
similar to how you'd use dependency injection in ASP.NET Core.
"""

import os
from openai import AzureOpenAI

from core.models import Document, RagResponse, RetrievedChunk
from core.chunker import TextChunker
from core.embedder import AzureEmbedder
from core.vector_store import InMemoryVectorStore


_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question
using ONLY the context passages provided below. If the answer is not in the
context, say "I don't have enough information to answer that."

Be concise and factual. Do not make things up."""


class RagPipeline:
    """
    End-to-end RAG pipeline.

    Typical lifecycle
    -----------------
    1. pipeline = RagPipeline()
    2. pipeline.ingest(documents)    <- one-time setup
    3. response = pipeline.query("...") <- call many times
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embedder = AzureEmbedder()
        self._store = InMemoryVectorStore()
        self._top_k = top_k

        self._chat_client = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_CHAT_ENDPOINT", os.environ["AZURE_OPENAI_ENDPOINT"]),
            api_key=os.environ.get("AZURE_CHAT_API_KEY", os.environ["AZURE_OPENAI_API_KEY"]),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        self._chat_deployment = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
    # ------------------------------------------------------------------
    # Ingestion — call this once when your documents change
    # ------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        """
        Chunk -> embed -> store all documents.
        """
        print(f"[RagPipeline] Ingesting {len(documents)} document(s)...")

        chunks = self._chunker.split_many(documents)
        print(f"[RagPipeline] Created {len(chunks)} chunk(s)")

        embedded = self._embedder.embed_chunks(chunks)
        self._store.add(embedded)

        print(f"[RagPipeline] Vector store now contains {self._store.count} chunk(s)")

    # ------------------------------------------------------------------
    # Querying — call this for every user question
    # ------------------------------------------------------------------

    def query(self, question: str) -> RagResponse:
        """
        1. Embed the question
        2. Retrieve the most relevant chunks
        3. Ask the LLM to answer using those chunks
        """
        # Step 1 — embed the question
        query_embedding = self._embedder.embed_text(question)

        # Step 2 — retrieve
        retrieved: list[RetrievedChunk] = self._store.search(
            query_embedding, top_k=self._top_k
        )

        if not retrieved:
            return RagResponse(
                answer="No documents have been ingested yet.",
                query=question,
                retrieved_chunks=[],
                model_used=self._chat_deployment,
            )

        # Step 3 — build context and call the LLM
        context = self._build_context(retrieved)

        completion = self._chat_client.chat.completions.create(
            model=self._chat_deployment,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )

        answer = completion.choices[0].message.content or ""

        return RagResponse(
            answer=answer,
            query=question,
            retrieved_chunks=retrieved,
            model_used=self._chat_deployment,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, rc in enumerate(chunks, start=1):
            source = rc.chunk.metadata.get("source", "unknown")
            parts.append(
                f"[{i}] (source: {source}, score: {rc.score:.2f})\n{rc.chunk.content}"
            )
        return "\n\n".join(parts)
