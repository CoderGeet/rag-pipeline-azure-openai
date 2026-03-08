"""
main.py — Entry point. Run this to try the RAG pipeline.

    python main.py

Make sure you have a .env file set up first.
"""

import uuid
from dotenv import load_dotenv

# Load .env before importing anything that reads os.environ
load_dotenv()

from core import Document, RagPipeline


def load_sample_documents() -> list[Document]:
    """Load the sample text file as a list of Documents."""
    with open("data/sample_docs.txt", encoding="utf-8") as f:
        raw = f.read()

    sections = [s.strip() for s in raw.split("---") if s.strip()]

    return [
        Document(
            id=str(uuid.uuid4()),
            content=section,
            source="sample_docs.txt",
        )
        for section in sections
    ]


def print_response(response) -> None:
    print("\n" + "=" * 60)
    print(f"Q: {response.query}")
    print("-" * 60)
    print(f"A: {response.answer}")
    print("-" * 60)
    print(f"Retrieved {len(response.retrieved_chunks)} chunk(s):")
    for i, rc in enumerate(response.retrieved_chunks, 1):
        print(f"  [{i}] score={rc.score:.3f}  -> \"{rc.chunk.content[:80]}...\"")
    print("=" * 60)


def main() -> None:
    # 1. Create the pipeline
    pipeline = RagPipeline(chunk_size=400, chunk_overlap=40, top_k=2)

    # 2. Ingest documents
    documents = load_sample_documents()
    pipeline.ingest(documents)

    # 3. Interactive loop
    print("\nRAG Pipeline ready. Type your question or 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        response = pipeline.query(question)
        print_response(response)


if __name__ == "__main__":
    main()
