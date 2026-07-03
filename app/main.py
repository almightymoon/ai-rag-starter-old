import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    yield


app = FastAPI(title="AI RAG Starter", version="1.0.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
async def health():
    return {"status": "ok", "vector_store": pipeline.collection_count if pipeline else 0}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    count = pipeline.ingest_text(req.text, req.source)
    return {"chunks_added": count, "source": req.source}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    content = (await file.read()).decode("utf-8", errors="replace")
    count = pipeline.ingest_text(content, file.filename or "upload")
    return {"chunks_added": count, "filename": file.filename}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not configured")
    answer, sources = pipeline.query(req.question, req.top_k)
    return QueryResponse(answer=answer, sources=sources)
