from dataclasses import dataclass, field

@dataclass
class Document:
    """Represents a source document loaded into the pipeline."""
    id: str
    content: str
    source: str = "" # Optional source information (e.g., file name, URL)
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a Document after splitting."""
    id: str
    document_id: str
    content: str
    index: int  # position within the parent document
    metadata: dict = field(default_factory=dict)

@dataclass
class EmbeddedChunk:
    """A Chunk paired with its embedding vector."""
    chunk: Chunk
    embedding: list[float]


@dataclass
class RetrievedChunk:
    """A Chunk returned from a similarity search, with its score."""
    chunk: Chunk
    score: float


@dataclass
class RagResponse:
    """The final response from the RAG pipeline."""
    answer: str
    query: str
    retrieved_chunks: list[RetrievedChunk]
    model_used: str