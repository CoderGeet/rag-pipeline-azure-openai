"""
inspect_vectors.py — Peek inside the FAISS vector store
after ingestion to see what got stored.
"""

import uuid
from dotenv import load_dotenv
load_dotenv()

from core import Document, RagPipeline
from core.vector_store import InMemoryVectorStore
from core.chunker import TextChunker
from core.embedder import AzureEmbedder


def load_sample_documents():
    with open("data/sample_docs.txt", encoding="utf-8") as f:
        raw = f.read()
    sections = [s.strip() for s in raw.split("---") if s.strip()]
    return [
        Document(id=str(uuid.uuid4()), content=section, source="sample_docs.txt")
        for section in sections
    ]


def main():
    # Step 1 — chunk the documents
    chunker = TextChunker(chunk_size=400, chunk_overlap=40)
    documents = load_sample_documents()
    chunks = chunker.split_many(documents)

    print(f"\n{'='*60}")
    print(f"CHUNKS ({len(chunks)} total)")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}")
        print(f"  ID          : {chunk.id}")
        print(f"  Document ID : {chunk.document_id}")
        print(f"  Index       : {chunk.index}")
        print(f"  Source      : {chunk.metadata.get('source')}")
        print(f"  Length      : {len(chunk.content)} chars")
        print(f"  Preview     : {chunk.content[:100]}...")

    # Step 2 — embed the chunks
    print(f"\n{'='*60}")
    print("EMBEDDING CHUNKS (calling Azure OpenAI)...")
    print(f"{'='*60}")
    embedder = AzureEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    print(f"\nEMBEDDINGS ({len(embedded_chunks)} total)")
    print(f"{'='*60}")
    for i, ec in enumerate(embedded_chunks):
        print(f"\nEmbedded Chunk {i}")
        print(f"  Text preview  : {ec.chunk.content[:80]}...")
        print(f"  Vector length : {len(ec.embedding)}")
        print(f"  First 5 values: {[round(v, 4) for v in ec.embedding[:5]]}")
        print(f"  Last  5 values: {[round(v, 4) for v in ec.embedding[-5:]]}")

    # Step 3 — store in FAISS and inspect
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    print(f"\n{'='*60}")
    print(f"VECTOR STORE")
    print(f"{'='*60}")
    print(f"  Total vectors stored : {store.count}")
    print(f"  Vector dimensions    : {store._index.d}")
    print(f"  Index type           : {type(store._index).__name__}")

    # Step 4 — run a test search and show scores
    print(f"\n{'='*60}")
    print("TEST SEARCH")
    print(f"{'='*60}")

    test_queries = [
        "return policy",
        "warranty coverage",
        "shipping time",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_vector = embedder.embed_text(query)
        results = store.search(query_vector, top_k=2)
        for j, result in enumerate(results):
            print(f"  [{j+1}] score={result.score:.4f}")
            print(f"       source={result.chunk.metadata.get('source')}")
            print(f"       text  ={result.chunk.content[:80]}...")


if __name__ == "__main__":
    main()
