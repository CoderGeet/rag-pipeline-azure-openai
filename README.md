# RAG Pipeline with Azure OpenAI

A Retrieval-Augmented Generation (RAG) pipeline built from scratch
in Python using Azure OpenAI — without using LangChain or any
abstraction framework.

## Status
✅ Local RAG pipeline — complete and working
✅ Azure AI Search — complete
✅ FastAPI REST API — complete
🔄 Azure App Service deployment — in progress   
⬜ Azure Function for scheduled ingestion — planned

## What it does
Answers natural language questions about your documents by:
1. Splitting documents into chunks
2. Converting chunks to embedding vectors via Azure OpenAI
3. Storing vectors in a FAISS index
4. Finding the most relevant chunks at query time
5. Passing them to GPT-4o to generate a grounded answer

## Architecture
```
User Query → Embed Query → Search Vector DB → Retrieve Chunks → LLM → Answer
```

## Project Structure
```
rag-demo/
├── core/
│   ├── models.py          # Data classes (Document, Chunk, RagResponse)
│   ├── chunker.py         # Splits documents into chunks
│   ├── embedder.py        # Azure OpenAI embeddings
│   ├── vector_store.py    # FAISS vector store
│   └── rag_pipeline.py    # Orchestrates everything
├── data/
│   └── sample_docs.txt    # Sample documents
├── main.py                # Interactive CLI entry point
├── inspect_vectors.py     # Tool to inspect vector store
└── requirements.txt
```

## Tech Stack
- Python 3.13
- Azure OpenAI (text-embedding-ada-002 + gpt-4o)
- FAISS (vector similarity search)
- python-dotenv (configuration)

## Setup
1. Clone the repo
```
git clone https://github.com/your-username/rag-pipeline-azure-openai.git
cd rag-pipeline-azure-openai
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Configure environment
```
copy .env.example .env
# Fill in your Azure OpenAI credentials
```

4. Run
```
python main.py
```

## Key Concepts Demonstrated
- RAG architecture built from scratch
- Azure OpenAI Embeddings and Chat Completions API
- Vector similarity search with FAISS
- Clean layered architecture
- Environment-based configuration

## Background
Built as a learning project to understand RAG pipelines deeply
before using abstraction frameworks like LangChain.
