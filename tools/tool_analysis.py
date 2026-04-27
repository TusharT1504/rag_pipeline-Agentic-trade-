"""Analysis helpers for retrieved tool results."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from config import settings
from observability import add_current_run_metadata, traceable


def _round_score(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


@traceable(run_type="tool", name="Retrieval Tool Analysis")
def tool_analysis_tool(query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retrieval quality and source coverage for a query."""
    scores = [
        float(chunk.get("score", 0.0))
        for chunk in retrieved_chunks
        if isinstance(chunk, dict)
    ]
    source_counts = Counter(
        chunk.get("metadata", {}).get("document_name", "unknown")
        for chunk in retrieved_chunks
        if isinstance(chunk, dict)
    )

    analysis = {
        "query": query,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "top_k": settings.TOP_K,
        "retrieved_count": len(retrieved_chunks),
        "score_summary": {
            "top": _round_score(max(scores) if scores else None),
            "average": _round_score(mean(scores) if scores else None),
            "lowest": _round_score(min(scores) if scores else None),
        },
        "source_coverage": dict(source_counts.most_common(10)),
        "top_chunks": [
            {
                "rank": index,
                "score": chunk.get("score"),
                "document_name": chunk.get("metadata", {}).get("document_name", "unknown"),
                "page_number": chunk.get("metadata", {}).get("page_number"),
                "section": chunk.get("metadata", {}).get("section", "General"),
            }
            for index, chunk in enumerate(retrieved_chunks[:5], start=1)
            if isinstance(chunk, dict)
        ],
    }

    add_current_run_metadata(
        {
            "retrieved_count": analysis["retrieved_count"],
            "retrieval_top_score": analysis["score_summary"]["top"],
            "retrieval_average_score": analysis["score_summary"]["average"],
            "retrieval_source_count": len(source_counts),
        }
    )
    return analysis
