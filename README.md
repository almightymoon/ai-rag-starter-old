# AI RAG Starter

Production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, and OpenAI-compatible LLMs.

[![CI](https://github.com/almightymoon/ai-rag-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/almightymoon/ai-rag-starter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features

- **Document ingestion** — PDF, TXT, Markdown via LangChain loaders
- **Vector store** — ChromaDB with persistent storage
- **Embeddings** — OpenAI or any OpenAI-compatible endpoint (Ollama, vLLM)
- **RAG pipeline** — Chunk → embed → retrieve → generate with citations
- **REST API** — `/ingest`, `/query`, `/health` endpoints
- **Docker** — One-command local deployment

## Architecture

```
Documents → Chunker → Embeddings → ChromaDB
                                      ↓
User Query → Embed → Retrieve top-k → LLM → Answer + Sources
```

## Quick Start

```bash
cp .env.example .env   # add OPENAI_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Kubernetes is a container orchestration platform.", "source": "docs"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Kubernetes?"}'
```

### Docker

```bash
docker compose up --build
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ingest` | POST | Add text to vector store |
| `/ingest/file` | POST | Upload and ingest a file |
| `/query` | POST | RAG question answering |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key for embeddings + LLM |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible API base URL |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Vector store path |

## License

MIT © [almightymoon](https://github.com/almightymoon)
