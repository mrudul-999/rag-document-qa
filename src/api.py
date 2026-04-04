# src/api.py

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingest import ingest
from src.query import load_vectorstore, build_qa_chain, query_documents

# ── REQUEST / RESPONSE MODELS (DTOs) ─────────────────────
# Interview point: these are exactly the DTOs you built at eQube-MI.
# Pydantic models validate input automatically — no manual checks needed.
# FastAPI generates OpenAPI docs from these automatically.

class QueryRequest(BaseModel):
    question: str = Field(
        ...,                              # required field
        min_length=3,
        max_length=500,
        description="The question to ask about the documents",
        example="What is the attention mechanism?"
    )
    k: int = Field(
        default=4,
        ge=1,                             # greater than or equal to 1
        le=10,                            # less than or equal to 10
        description="Number of chunks to retrieve"
    )

from typing import Union

class SourceChunk(BaseModel):
    content: str
    page: Union[int, str]
    source: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    num_chunks_used: int
    response_time_ms: float

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    index_exists: bool
    model: str


# ── APP STATE ─────────────────────────────────────────────
# Store loaded models in app state so they're loaded ONCE at startup
# not on every request — critical for performance
app_state = {}


# ── LIFESPAN — runs on startup and shutdown ───────────────
# Interview point: @asynccontextmanager lifespan replaces the old
# @app.on_event("startup") pattern — it's the modern FastAPI way.
# Loading models at startup means the first request isn't slow.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("🚀 Starting RAG API...")
    if Path("faiss_index").exists():
        print("   Loading FAISS index and building QA chain...")
        vectorstore = load_vectorstore()
        qa_chain = build_qa_chain(vectorstore)
        app_state["qa_chain"] = qa_chain
        app_state["ready"] = True
        print("   ✅ Ready")
    else:
        print("   ⚠️  No FAISS index found — upload a document first")
        app_state["ready"] = False

    yield  # app runs here

    # SHUTDOWN
    print("Shutting down...")
    app_state.clear()


# ── APP INIT ──────────────────────────────────────────────
app = FastAPI(
    title="Document Q&A with RAG",
    description="Upload PDFs and ask natural language questions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
# In production you'd whitelist specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── ENDPOINTS ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Convention: always have /health — monitoring systems ping this.
    Returns 200 if up, includes whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        index_exists=Path("faiss_index").exists(),
        model="sentence-transformers/all-MiniLM-L6-v2",
    )


@app.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and trigger ingestion pipeline.
    Saves file to disk then runs full ingest() pipeline.

    Interview point: UploadFile streams the file — memory efficient
    for large PDFs. We save to disk first because PyPDFLoader
    needs a file path, not a file object.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Save uploaded file
    file_path = f"data/{file.filename}"
    Path("data").mkdir(exist_ok=True)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Run ingestion pipeline
    try:
        vectorstore = ingest(file_path)
        chunk_count = vectorstore.index.ntotal  # FAISS internal count

        # Rebuild QA chain with new index
        qa_chain = build_qa_chain(vectorstore)
        app_state["qa_chain"] = qa_chain
        app_state["ready"] = True

        return IngestResponse(
            message="Document ingested successfully",
            filename=file.filename,
            chunks_created=chunk_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question about the uploaded documents.

    Measures response time — important for your resume metric
    ('sub-100ms retrieval') — retrieval is fast, LLM call is slower.
    """
    if not app_state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="No documents ingested yet. Upload a PDF first via /upload"
        )

    start_time = time.time()

    try:
        result = query_documents(app_state["qa_chain"], request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_time_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        num_chunks_used=result["num_chunks_used"],
        response_time_ms=round(response_time_ms, 2),
    )


@app.get("/")
async def root():
    return {
        "message": "Document Q&A RAG API",
        "docs": "/docs",           # FastAPI auto-generated Swagger UI
        "health": "/health",
    }