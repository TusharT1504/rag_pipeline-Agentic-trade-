"""api.py — Optional FastAPI wrapper for the LangGraph RAG system.

Start with:
    uvicorn api:app --reload --port 8000

Then query:
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"query": "What was the export growth in 2023?"}'
"""

from __future__ import annotations
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from main import run_query

app = FastAPI(
    title="LangGraph RAG API",
    description="PDF-backed RAG system with Pinecone + LangGraph",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class SourceItem(BaseModel):
    document_name: str
    page_number: int
    section: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: str
    tool_analysis: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
def validate_settings():
    settings.validate()


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = run_query(request.query, top_k=request.top_k)
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", "low"),
        tool_analysis=result.get("tool_analysis", {}),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
