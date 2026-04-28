"""
API Layer — FastAPI application exposing the RAG system.

Endpoints:
  POST /query       → Retrieval mode
  POST /ingest      → Ingestion mode
  GET  /health      → Health check
  GET  /namespaces  → List configured namespaces
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import get_settings
from graph.rag_graph import get_rag_graph
from graph.state import RAGState, get_state_value
from observability.langsmith import traced_function

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the graph and embedding model at startup."""
    logger.info("Starting RAG API — pre-warming graph and embedding model…")
    # Import triggers singleton initialisation (model load)
    from tools.st_model import get_embedding_model  # noqa: F401
    get_embedding_model()
    get_rag_graph()
    logger.info("RAG API ready.")
    yield
    logger.info("RAG API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Production RAG System",
    description=(
        "LangGraph + Pinecone + SentenceTransformers + Groq — "
        "Retrieval-Augmented Generation API"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str = Field(..., min_length=3, description="The user's question.")
    namespaces: list[str] = Field(
        default_factory=list,
        description="Pinecone namespaces to search. Defaults to all configured namespaces.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    sector: str = Field(
        default="",
        description=(
            "Sector label injected into the YAML prompt as sector_input "
            "(e.g. 'ESDM', 'cement'). Auto-derived from retrieved chunks when omitted."
        ),
    )


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    query: str
    answer: str
    namespaces_searched: list[str]
    retrieved_docs_count: int
    elapsed_ms: float
    error: str = ""


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    pdf_paths: list[str] = Field(..., min_length=1, description="Absolute paths to PDF files.")
    sector: str = Field(..., description="Sector / Pinecone namespace for the documents.")


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    sector: str
    upserted_count: int
    elapsed_ms: float
    error: str = ""


class HealthResponse(BaseModel):
    status: str
    model: str
    index: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
@traced_function("health_check", metadata={"entrypoint": "api"})
async def health_check() -> HealthResponse:
    """Return a simple liveness probe."""
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        index=settings.pinecone_index_name,
    )


@app.get("/namespaces", tags=["System"])
@traced_function("list_namespaces", metadata={"entrypoint": "api"})
async def list_namespaces() -> dict[str, Any]:
    """Return the configured default namespaces."""
    return {"default_namespaces": settings.default_namespaces}


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
@traced_function("query_endpoint", metadata={"entrypoint": "api", "mode": "query"})
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Retrieval mode — answer a question using context from Pinecone.

    The embedding model and Pinecone are queried; no PDF processing occurs.
    """
    start = time.perf_counter()
    namespaces = request.namespaces or settings.default_namespaces

    logger.info(
        "POST /query | query='%s' namespaces=%s top_k=%d",
        request.query[:80],
        namespaces,
        request.top_k,
    )

    initial_state = RAGState(
        ingest_flag=False,
        query=request.query,
        namespaces=namespaces,
        top_k=request.top_k,
        sector=request.sector,
    )

    try:
        graph = get_rag_graph()
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph invocation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal graph error: {exc}",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        query=request.query,
        answer=get_state_value(final_state, "answer", ""),
        namespaces_searched=namespaces,
        retrieved_docs_count=len(get_state_value(final_state, "retrieved_chunks", [])),
        elapsed_ms=round(elapsed_ms, 2),
        error=get_state_value(final_state, "error", ""),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
@traced_function("ingest_endpoint", metadata={"entrypoint": "api", "mode": "ingest"})
async def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    """
    Ingestion mode — load PDFs, chunk, embed, and upsert into Pinecone.

    This endpoint is ONLY used when new documents need to be indexed.
    Normal queries should use POST /query.
    """
    start = time.perf_counter()
    logger.info(
        "POST /ingest | sector='%s' files=%d",
        request.sector,
        len(request.pdf_paths),
    )

    initial_state = RAGState(
        ingest_flag=True,
        pdf_paths=request.pdf_paths,
        sector=request.sector,
    )

    try:
        graph = get_rag_graph()
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Ingestion graph failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {exc}",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    error = get_state_value(final_state, "error", "")
    if error:
        logger.error("Ingestion completed with error: %s", error)

    return IngestResponse(
        sector=request.sector,
        upserted_count=get_state_value(final_state, "upserted_count", 0),
        elapsed_ms=round(elapsed_ms, 2),
        error=error,
    )
