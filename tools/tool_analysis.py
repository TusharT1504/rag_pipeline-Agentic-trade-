"""
Tool Analysis Utility.
Provides helpers to inspect and summarise the outputs of pipeline tools,
useful for debugging, observability dashboards, and test assertions.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from graph.state import DocumentChunk, RetrievedChunk
from observability.langsmith import traced_function

logger = logging.getLogger(__name__)


# ── Ingestion analysis ────────────────────────────────────────────────────────


@traced_function("analyse_chunks", metadata={"component": "analysis"})
def analyse_chunks(chunks: list[DocumentChunk]) -> dict[str, Any]:
    """
    Summarise a list of ``DocumentChunk`` objects.

    Returns a dict with:
    - total_chunks
    - chunks_per_source (Counter)
    - chunks_per_page (Counter)
    - avg_char_length
    - min_char_length
    - max_char_length
    """
    if not chunks:
        return {"total_chunks": 0}

    lengths = [len(c.text) for c in chunks]
    per_source = Counter(c.source_file for c in chunks)
    per_page = Counter(c.page_number for c in chunks)

    summary = {
        "total_chunks": len(chunks),
        "chunks_per_source": dict(per_source),
        "chunks_per_page": dict(per_page),
        "avg_char_length": round(sum(lengths) / len(lengths), 1),
        "min_char_length": min(lengths),
        "max_char_length": max(lengths),
    }
    logger.debug("Chunk analysis: %s", summary)
    return summary


@traced_function("analyse_embeddings", metadata={"component": "analysis"})
def analyse_embeddings(vectors: list[list[float]]) -> dict[str, Any]:
    """
    Basic statistics on a batch of embedding vectors.

    Returns a dict with:
    - count
    - dimension
    - all_same_dimension (bool)
    """
    if not vectors:
        return {"count": 0}

    dims = [len(v) for v in vectors]
    summary = {
        "count": len(vectors),
        "dimension": dims[0] if dims else 0,
        "all_same_dimension": len(set(dims)) == 1,
    }
    logger.debug("Embedding analysis: %s", summary)
    return summary


# ── Retrieval analysis ────────────────────────────────────────────────────────


@traced_function("analyse_retrieved_chunks", metadata={"component": "analysis"})
def analyse_retrieved_chunks(chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """
    Summarise Pinecone retrieval results.

    Returns a dict with:
    - total_retrieved
    - namespaces_hit (list)
    - score_stats (min, max, avg)
    - sources (list of unique source files)
    """
    if not chunks:
        return {"total_retrieved": 0, "namespaces_hit": [], "sources": []}

    scores = [c.score for c in chunks]
    summary = {
        "total_retrieved": len(chunks),
        "namespaces_hit": list({c.namespace for c in chunks}),
        "score_stats": {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "avg": round(sum(scores) / len(scores), 4),
        },
        "sources": list({c.source_file for c in chunks}),
    }
    logger.debug("Retrieval analysis: %s", summary)
    return summary


@traced_function("format_context_for_display", metadata={"component": "analysis"})
def format_context_for_display(chunks: list[RetrievedChunk], max_chars: int = 200) -> str:
    """
    Return a human-readable summary of retrieved chunks, e.g. for logging.

    Args:
        chunks: Retrieved chunks.
        max_chars: Maximum characters of chunk text to display per entry.

    Returns:
        Multi-line string.
    """
    lines = [f"Retrieved {len(chunks)} chunk(s):"]
    for i, c in enumerate(chunks, 1):
        preview = c.text[:max_chars].replace("\n", " ")
        if len(c.text) > max_chars:
            preview += "…"
        lines.append(
            f"  [{i}] ns={c.namespace} | src={c.source_file}:p{c.page_number}"
            f" | score={c.score:.4f}\n      {preview}"
        )
    return "\n".join(lines)
