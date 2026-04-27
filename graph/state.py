"""graph/state.py — LangGraph shared state schema"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class RAGState(BaseModel):
    """
    Shared mutable state that flows through every node in the LangGraph.
    Each node reads what it needs and writes its outputs back here.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    query: str = ""

    # ── Embedding pipeline ─────────────────────────────────────────────────
    embeddings_exist: Optional[bool] = None      # None = not yet checked
    raw_texts: list[dict[str, Any]] = Field(default_factory=list)
    # Each item: {"document_name": str, "page_number": int, "text": str}

    chunks: list[dict[str, Any]] = Field(default_factory=list)
    # Each item: {"id": str, "text": str, "metadata": dict}

    vectors_stored: bool = False

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    # Each item: {"text": str, "score": float, "metadata": dict}

    # ── Answer generation ──────────────────────────────────────────────────
    tool_analysis: dict[str, Any] = Field(default_factory=dict)
    # Retrieval diagnostics for tracing/debugging.

    final_answer: Optional[dict[str, Any]] = None
    # Shape: {"answer": str, "sources": [...], "confidence": str}

    # ── Control / error ────────────────────────────────────────────────────
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
