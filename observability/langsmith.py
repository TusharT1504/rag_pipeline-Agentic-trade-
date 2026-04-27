"""LangSmith tracing utilities.

The project should still run before dependencies are installed or when tracing
is disabled, so this module degrades to no-op decorators if LangSmith is absent.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from langsmith import get_current_run_tree, traceable
except ImportError:  # pragma: no cover - exercised only without optional deps
    get_current_run_tree = None

    def traceable(*args: Any, **kwargs: Any) -> Callable:
        """No-op replacement with the same decorator ergonomics."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable) -> Callable:
            return func

        return decorator


def _text_preview(value: str, limit: int = 180) -> str:
    return value.replace("\n", " ").strip()[:limit]


def _get_sequence(inputs: dict[str, Any], key: str) -> list[Any]:
    value = inputs.get(key)
    return value if isinstance(value, list) else []


def add_current_run_metadata(metadata: dict[str, Any]) -> None:
    """Attach diagnostic metadata to the active LangSmith span if one exists."""
    if get_current_run_tree is None:
        return

    try:
        run_tree = get_current_run_tree()
        if run_tree is None:
            return
        run_tree.metadata.update(metadata)
    except Exception as exc:  # pragma: no cover - tracing must not break RAG
        logger.debug("Could not update LangSmith run metadata: %s", exc)


def compact_pages_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    pages = _get_sequence(inputs, "pages")
    return {
        "page_count": len(pages),
        "documents": sorted(
            {page.get("document_name", "unknown") for page in pages if isinstance(page, dict)}
        )[:20],
    }


def compact_pages_outputs(output: Any) -> dict[str, Any]:
    pages = output if isinstance(output, list) else []
    document_counts = Counter(
        page.get("document_name", "unknown")
        for page in pages
        if isinstance(page, dict)
    )
    return {
        "page_count": len(pages),
        "document_count": len(document_counts),
        "documents": dict(document_counts.most_common(20)),
    }


def compact_embedding_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    chunks = _get_sequence(inputs, "chunks")
    return {
        "chunk_count": len(chunks),
        "sample_chunk_ids": [
            chunk.get("id")
            for chunk in chunks[:5]
            if isinstance(chunk, dict) and chunk.get("id")
        ],
        "sample_text": [
            _text_preview(chunk.get("text", ""))
            for chunk in chunks[:3]
            if isinstance(chunk, dict)
        ],
    }


def compact_embedding_outputs(output: Any) -> dict[str, Any]:
    chunks = output if isinstance(output, list) else []
    first_embedding = None
    for chunk in chunks:
        if isinstance(chunk, dict):
            first_embedding = chunk.get("embedding")
            if first_embedding is not None:
                break

    return {
        "chunk_count": len(chunks),
        "embedding_dimension": len(first_embedding) if isinstance(first_embedding, list) else None,
        "sample_chunk_ids": [
            chunk.get("id")
            for chunk in chunks[:5]
            if isinstance(chunk, dict) and chunk.get("id")
        ],
    }


def compact_search_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    embedding = inputs.get("query_embedding")
    return {
        "query_embedding_dimension": len(embedding) if isinstance(embedding, list) else None,
        "top_k": inputs.get("top_k"),
    }


def compact_search_outputs(output: Any) -> dict[str, Any]:
    chunks = output if isinstance(output, list) else []
    scores = [
        float(chunk.get("score", 0.0))
        for chunk in chunks
        if isinstance(chunk, dict)
    ]
    return {
        "result_count": len(chunks),
        "top_score": max(scores) if scores else None,
        "lowest_score": min(scores) if scores else None,
        "sources": [
            {
                "document_name": chunk.get("metadata", {}).get("document_name", "unknown"),
                "page_number": chunk.get("metadata", {}).get("page_number"),
                "score": chunk.get("score"),
            }
            for chunk in chunks[:10]
            if isinstance(chunk, dict)
        ],
    }


def retrieved_chunks_as_documents(output: Any) -> list[dict[str, Any]]:
    """Format retrieved chunks for LangSmith's retriever UI."""
    chunks = output if isinstance(output, list) else []
    documents: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        metadata = dict(chunk.get("metadata", {}))
        metadata["score"] = chunk.get("score")
        documents.append(
            {
                "page_content": chunk.get("text", ""),
                "type": "Document",
                "metadata": metadata,
            }
        )

    return documents
