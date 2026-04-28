"""
Vector Store Tool — Pinecone integration layer.

Responsibilities:
  - UPSERT vectors during ingestion mode
  - QUERY vectors during retrieval mode

This module is the ONLY place in the codebase that communicates
directly with Pinecone.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config.settings import get_settings
from graph.state import DocumentChunk, RetrievedChunk
from observability.langsmith import traced_function, traced_tool

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Pinecone client singleton ────────────────────────────────────────────────

_pc: Pinecone | None = None
_index = None  # pinecone.Index


@traced_function("pinecone_get_client", metadata={"component": "pinecone"})
def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        logger.info("Initialising Pinecone client…")
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


@traced_function("pinecone_get_index", metadata={"component": "pinecone"})
def _get_index():
    """Return (and lazily create) the Pinecone Index object."""
    global _index
    if _index is not None:
        return _index

    pc = _get_client()
    index_name = settings.pinecone_index_name

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s'…", index_name)
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
        # Wait for the index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            logger.debug("Waiting for index to be ready…")
            time.sleep(2)
        logger.info("Index '%s' created and ready.", index_name)
    else:
        logger.debug("Using existing Pinecone index '%s'.", index_name)

    _index = pc.Index(index_name)
    return _index


# ── Upsert (ingestion mode) ─────────────────────────────────────────────────


@traced_tool("upsert_chunks", metadata={"pipeline": "ingestion", "stage": "vector_store"})
def upsert_chunks(
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
    batch_size: int = 100,
) -> int:
    """
    Upsert document chunks and their embeddings into Pinecone.

    Each chunk is stored in the namespace derived from its ``sector`` field.
    Metadata stored per vector:
      - sector
      - source_file
      - chunk_id
      - page_number
      - text (truncated to 1 000 chars to stay within Pinecone metadata limits)

    Args:
        chunks: DocumentChunk objects to upsert.
        vectors: Parallel list of embedding vectors.
        batch_size: Number of vectors per upsert call.

    Returns:
        Total number of vectors upserted.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors."
        )

    index = _get_index()
    total_upserted = 0

    # Group by namespace (sector)
    by_namespace: dict[str, list[tuple[DocumentChunk, list[float]]]] = {}
    for chunk, vec in zip(chunks, vectors):
        by_namespace.setdefault(chunk.sector, []).append((chunk, vec))

    for namespace, items in by_namespace.items():
        records = [
            {
                "id": chunk.chunk_id,
                "values": vec,
                "metadata": {
                    "sector": chunk.sector,
                    "source_file": chunk.source_file,
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "text": chunk.text[:1000],
                },
            }
            for chunk, vec in items
        ]

        # Batch upsert
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            logger.debug(
                "Upserted batch of %d to namespace '%s'", len(batch), namespace
            )

    logger.info("Total vectors upserted: %d", total_upserted)
    return total_upserted


# ── Query (retrieval mode) ──────────────────────────────────────────────────


def _metadata_value(meta: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return default


def _page_number(meta: dict[str, Any]) -> int:
    raw_page = _metadata_value(meta, "page_number", "page", default=0)
    try:
        return int(raw_page)
    except (TypeError, ValueError):
        return 0


def _sort_key(vector_id: str) -> list[int | str]:
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", vector_id):
        if not part:
            continue
        parts.append(int(part) if part.isdigit() else part)
    return parts


def _to_retrieved_chunk(
    vector_id: str,
    metadata: dict[str, Any],
    namespace: str,
    score: float = 1.0,
) -> RetrievedChunk:
    text = _metadata_value(metadata, "text", "chunk_text", "content")
    return RetrievedChunk(
        chunk_id=vector_id,
        text=str(text or ""),
        score=score,
        sector=str(_metadata_value(metadata, "sector", "source", default=namespace)),
        source_file=str(
            _metadata_value(metadata, "source_file", "document_name", "source", default="")
        ),
        page_number=_page_number(metadata),
        namespace=namespace,
        metadata=metadata,
    )


@traced_tool("query_namespace", metadata={"pipeline": "retrieval", "stage": "vector_store"})
def query_namespace(
    query_vector: list[float],
    namespace: str,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Query a single Pinecone namespace.

    Args:
        query_vector: Dense embedding of the user query.
        namespace: Pinecone namespace to search.
        top_k: Maximum number of results to return.
        score_threshold: Minimum cosine similarity score; below this, chunks
            are discarded. Defaults to settings value.

    Returns:
        List of ``RetrievedChunk`` objects sorted by descending score.
    """
    threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
    index = _get_index()

    logger.debug("Querying namespace='%s' top_k=%d", namespace, top_k)
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    results: list[RetrievedChunk] = []
    for match in response.get("matches", []):
        score = match.get("score", 0.0)
        if score < threshold:
            logger.debug("Dropping match score=%.4f (below threshold %.4f)", score, threshold)
            continue
        meta: dict[str, Any] = match.get("metadata", {})
        results.append(_to_retrieved_chunk(match["id"], meta, namespace, score=score))

    logger.debug("Namespace '%s' → %d chunk(s) above threshold", namespace, len(results))
    return results


@traced_tool("query_namespaces", metadata={"pipeline": "retrieval", "stage": "vector_store"})
def query_namespaces(
    query_vector: list[float],
    namespaces: list[str],
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Query multiple Pinecone namespaces and merge results.

    Results are de-duplicated by chunk_id and re-ranked by score
    before being truncated to *top_k*.

    Args:
        query_vector: Dense embedding of the user query.
        namespaces: Namespaces to query (multi-namespace retrieval).
        top_k: Final number of results after merging all namespaces.
        score_threshold: Minimum score filter.

    Returns:
        Top-k ``RetrievedChunk`` objects across all namespaces.
    """
    all_results: list[RetrievedChunk] = []
    seen_ids: set[str] = set()

    for ns in namespaces:
        ns_results = query_namespace(
            query_vector=query_vector,
            namespace=ns,
            top_k=top_k,  # fetch top_k per namespace; trim after merge
            score_threshold=score_threshold,
        )
        for chunk in ns_results:
            if chunk.chunk_id not in seen_ids:
                all_results.append(chunk)
                seen_ids.add(chunk.chunk_id)

    # Re-rank globally and truncate
    all_results.sort(key=lambda c: c.score, reverse=True)
    final = all_results[:top_k]
    logger.info(
        "Multi-namespace query across %d namespace(s) → %d result(s)",
        len(namespaces),
        len(final),
    )
    return final


@traced_tool("fetch_namespace_chunks", metadata={"pipeline": "retrieval", "stage": "vector_store"})
def fetch_namespace_chunks(
    namespace: str,
    max_records: int | None = None,
    batch_size: int = 100,
) -> list[RetrievedChunk]:
    """
    Fetch stored chunks directly from a Pinecone namespace without query embedding.

    This lists vector IDs in the namespace and fetches their metadata. It supports
    both the current ingestion metadata keys and older records that used
    ``chunk_text`` / ``document_name`` / ``source``.
    """
    index = _get_index()
    limit = max_records if max_records is not None else settings.namespace_fetch_limit
    limit = limit if limit and limit > 0 else None

    ids: list[str] = []
    pagination_token: str | None = None

    while True:
        page_limit = min(batch_size, limit - len(ids)) if limit else batch_size
        response = index.list_paginated(
            namespace=namespace,
            limit=page_limit,
            pagination_token=pagination_token,
        )
        vectors = response.get("vectors", []) if hasattr(response, "get") else response.vectors
        ids.extend(vector["id"] if isinstance(vector, dict) else vector.id for vector in vectors)

        if limit and len(ids) >= limit:
            ids = ids[:limit]
            break

        pagination = response.get("pagination", {}) if hasattr(response, "get") else response.pagination
        pagination_token = None
        if pagination:
            pagination_token = (
                pagination.get("next")
                if isinstance(pagination, dict)
                else getattr(pagination, "next", None)
            )
        if not pagination_token:
            break

    ids = sorted(set(ids), key=_sort_key)
    logger.info("Listed %d vector ID(s) from namespace '%s'", len(ids), namespace)

    chunks: list[RetrievedChunk] = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        fetched = index.fetch(ids=batch_ids, namespace=namespace)
        vectors = fetched.get("vectors", {}) if hasattr(fetched, "get") else fetched.vectors
        for vector_id in batch_ids:
            vector = vectors.get(vector_id)
            if vector is None:
                continue
            metadata = vector.get("metadata", {}) if isinstance(vector, dict) else vector.metadata
            chunks.append(_to_retrieved_chunk(vector_id, metadata or {}, namespace))

    logger.info("Fetched %d chunk(s) from namespace '%s'", len(chunks), namespace)
    return chunks


@traced_tool("fetch_namespaces_chunks", metadata={"pipeline": "retrieval", "stage": "vector_store"})
def fetch_namespaces_chunks(
    namespaces: list[str],
    max_records_per_namespace: int | None = None,
) -> list[RetrievedChunk]:
    """Fetch stored chunks from multiple namespaces without vector search."""
    chunks: list[RetrievedChunk] = []
    seen_ids: set[tuple[str, str]] = set()

    for namespace in namespaces:
        for chunk in fetch_namespace_chunks(namespace, max_records=max_records_per_namespace):
            key = (chunk.namespace, chunk.chunk_id)
            if key in seen_ids:
                continue
            chunks.append(chunk)
            seen_ids.add(key)

    logger.info(
        "Fetched %d chunk(s) across %d namespace(s)", len(chunks), len(namespaces)
    )
    return chunks
