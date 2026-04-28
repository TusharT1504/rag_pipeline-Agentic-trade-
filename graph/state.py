"""
Typed state definition for the RAG LangGraph.
A single RAGState dataclass flows through every node; each node
mutates only the fields it owns and returns the updated state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a single chunk of text derived from a source document."""

    chunk_id: str
    text: str
    sector: str
    source_file: str
    page_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned from Pinecone retrieval with a similarity score."""

    chunk_id: str
    text: str
    score: float
    sector: str
    source_file: str
    page_number: int
    namespace: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGState:
    """
    Shared mutable state that flows through every LangGraph node.

    Ingestion fields are populated only in ingestion mode.
    Retrieval fields are populated only in retrieval mode.
    Both modes share the ``error`` and ``mode`` fields.
    """

    # ── Mode control ─────────────────────────────────────────────────────────
    ingest_flag: bool = False
    """If True the graph runs the ingestion pipeline; else retrieval."""

    mode: str = "retrieval"
    """Human-readable mode label – 'retrieval' or 'ingestion'."""

    # ── Query / retrieval fields ──────────────────────────────────────────────
    query: str = ""
    """The user's natural-language question."""

    namespaces: list[str] = field(default_factory=list)
    """Pinecone namespaces to query (e.g. ['ESDM', 'cement'])."""

    top_k: int = 5
    """Number of nearest-neighbour chunks to retrieve."""

    query_embedding: list[float] = field(default_factory=list)
    """Dense vector representation of the query (populated by embedding node)."""

    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    """Chunks retrieved from Pinecone (populated by retrieval node)."""

    answer: str = ""
    """Final generated answer (populated by answer-generation node)."""

    # ── Ingestion fields ──────────────────────────────────────────────────────
    pdf_paths: list[str] = field(default_factory=list)
    """Absolute paths to PDF files to ingest."""

    sector: str = ""
    """Sector / namespace label for the ingested documents."""

    raw_documents: list[dict[str, Any]] = field(default_factory=list)
    """Raw page-level content dicts from the PDF loader node."""

    document_chunks: list[DocumentChunk] = field(default_factory=list)
    """Text chunks produced by the chunking node."""

    chunk_embeddings: list[list[float]] = field(default_factory=list)
    """Dense vectors for each document chunk (parallel to document_chunks)."""

    upserted_count: int = 0
    """Number of vectors successfully upserted to Pinecone."""

    # ── Shared ────────────────────────────────────────────────────────────────
    error: str = ""
    """Non-empty string signals a node-level error; downstream nodes should check."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata bag for observability / debugging."""


def get_state_value(state: RAGState | dict[str, Any], field_name: str, default: Any = None) -> Any:
    """Read a state field from either a RAGState object or LangGraph's dict output."""
    if isinstance(state, dict):
        return state.get(field_name, default)
    return getattr(state, field_name, default)
