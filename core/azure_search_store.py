"""
azure_search_store.py — Production vector store using Azure AI Search.

Drop-in replacement for InMemoryVectorStore — same add() and search()
interface, but vectors are stored persistently in Azure AI Search
instead of in-memory FAISS.
"""

import os
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential
from core.models import EmbeddedChunk, RetrievedChunk, Chunk


class AzureSearchVectorStore:
    """
    Production vector store backed by Azure AI Search.

    Same interface as InMemoryVectorStore:
        store = AzureSearchVectorStore()
        store.add(embedded_chunks)
        results = store.search(query_embedding, top_k=3)
    """

    def __init__(self):
        endpoint        = os.environ["AZURE_SEARCH_ENDPOINT"]
        api_key         = os.environ["AZURE_SEARCH_API_KEY"]
        self._index_name = os.environ.get("AZURE_SEARCH_INDEX", "rag-chunks")

        credential = AzureKeyCredential(api_key)

        self._index_client  = SearchIndexClient(endpoint, credential)
        self._search_client = SearchClient(endpoint, self._index_name, credential)

        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """
        Create the search index if it does not already exist.
        Like running a database migration — safe to call multiple times.
        """
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="rag-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="rag-algo")
            ],
            profiles=[
                VectorSearchProfile(
                    name="rag-profile",
                    algorithm_configuration_name="rag-algo",
                )
            ],
        )

        index = SearchIndex(
            name=self._index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self._index_client.create_or_update_index(index)
        print(f"[AzureSearchVectorStore] Index '{self._index_name}' ready")

    @property
    def count(self) -> int:
        """Return total number of documents in the index."""
        result = self._search_client.get_document_count()
        return result

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Upload embedded chunks to Azure AI Search.
        Safe to call multiple times — existing documents are updated.
        """
        if not embedded_chunks:
            return

        documents = [
            {
                "id":          ec.chunk.id,
                "document_id": ec.chunk.document_id,
                "content":     ec.chunk.content,
                "source":      ec.chunk.metadata.get("source", ""),
                "chunk_index": ec.chunk.index,
                "embedding":   ec.embedding,
            }
            for ec in embedded_chunks
        ]

        self._search_client.upload_documents(documents)
        print(f"[AzureSearchVectorStore] Uploaded {len(documents)} chunk(s)")

    def search(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[RetrievedChunk]:
        """
        Find the top_k most similar chunks using vector similarity search.
        """
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        results = self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k,
        )

        retrieved = []
        for r in results:
            retrieved.append(RetrievedChunk(
                chunk=Chunk(
                    id=r["id"],
                    document_id=r["document_id"],
                    content=r["content"],
                    index=r["chunk_index"],
                    metadata={"source": r["source"]},
                ),
                score=r["@search.score"],
            ))

        return retrieved
