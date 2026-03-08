"""
api.py — FastAPI REST API wrapper around the RAG pipeline.

Endpoints:
    GET  /health        — health check
    POST /ingest        — load documents into the pipeline
    POST /query         — ask a question and get an answer

Run locally:
    uvicorn api:app --reload
"""

import uuid
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from core import Document, RagPipeline


# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Pipeline API",
    description="A RAG pipeline built from scratch using Azure OpenAI and Azure AI Search",
    version="1.0.0",
)

# Single pipeline instance shared across all requests
# equivalent to a singleton service in ASP.NET Core
pipeline = RagPipeline(chunk_size=400, chunk_overlap=40, top_k=3)


# ── Request / Response models ─────────────────────────────────────────
# These are like DTOs in ASP.NET Core — they define the shape of
# request and response bodies

class IngestRequest(BaseModel):
    content: str
    source: str = "api"


class IngestResponse(BaseModel):
    status: str
    source: str
    chunks_stored: int


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: list[str]
    chunks_retrieved: int


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — used by Azure App Service to verify the app is running."""
    return {
        "status": "ok",
        "vector_store": os.environ.get("VECTOR_STORE", "local"),
        "chat_deployment": os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Load a document into the pipeline.

    Example request body:
    {
        "content": "Returns accepted within 30 days...",
        "source": "sharepoint:HR-Policy.docx"
    }
    """
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    doc = Document(
        id=str(uuid.uuid4()),
        content=request.content,
        source=request.source,
    )

    pipeline.ingest([doc])

    return IngestResponse(
        status="ingested",
        source=request.source,
        chunks_stored=pipeline._store.count,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a question and get an answer grounded in your documents.

    Example request body:
    {
        "question": "What is the return policy?"
    }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    response = pipeline.query(request.question)

    return QueryResponse(
        answer=response.answer,
        query=response.query,
        sources=[rc.chunk.metadata.get("source", "") for rc in response.retrieved_chunks],
        chunks_retrieved=len(response.retrieved_chunks),
    )


@app.post("/ingest/file")
def ingest_file():
    """
    Placeholder for file ingestion — extend this to accept
    uploaded PDFs or text files in a future iteration.
    """
    return {"status": "not implemented yet"}
