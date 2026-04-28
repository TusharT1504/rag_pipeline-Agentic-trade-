"""
Retrieval Tool — Retrieval pipeline, Stage 2.
Queries Pinecone using a pre-computed query embedding.
This module NEVER embeds documents or writes to Pinecone.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from graph.state import RetrievedChunk
from observability.langsmith import traced_tool
from tools.vector_store_tool import fetch_namespaces_chunks, query_namespaces

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrievalError(Exception):
    """Raised when retrieval fails or returns no usable results."""


@traced_tool("retrieve_chunks", metadata={"pipeline": "retrieval", "stage": "retrieval"})
def retrieve_chunks(
    query_embedding: list[float],
    namespaces: list[str],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks from Pinecone.

    Supports multi-namespace retrieval: results from all *namespaces* are
    merged and re-ranked by cosine similarity before truncating to *top_k*.

    Implements a fallback: if no results exceed the score threshold,
    the threshold is lowered to 0.0 and the query is retried once so
    that *some* context is always returned to the LLM.

    Args:
        query_embedding: Dense vector from the embedding stage.
        namespaces: Pinecone namespace(s) to search.
        top_k: Number of results to return; defaults to ``settings.retrieval_top_k``.
        score_threshold: Minimum similarity score; defaults to settings value.

    Returns:
        List of ``RetrievedChunk`` objects, sorted by descending score.

    Raises:
        RetrievalError: If *namespaces* is empty or query_embedding is empty.
    """
    if not namespaces:
        raise RetrievalError("At least one namespace must be specified for retrieval.")
    if not query_embedding:
        raise RetrievalError("query_embedding is empty — run the embedding node first.")

    k = top_k or settings.retrieval_top_k
    threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

    logger.info(
        "Retrieving top-%d chunks from namespace(s)=%s (threshold=%.3f)",
        k,
        namespaces,
        threshold,
    )

    chunks = query_namespaces(
        query_vector=query_embedding,
        namespaces=namespaces,
        top_k=k,
        score_threshold=threshold,
    )

    # Fallback: retry with zero threshold if nothing came back
    if not chunks and threshold > 0.0:
        logger.warning(
            "No chunks above threshold %.3f — retrying with threshold=0.0 (fallback).",
            threshold,
        )
        chunks = query_namespaces(
            query_vector=query_embedding,
            namespaces=namespaces,
            top_k=k,
            score_threshold=0.0,
        )

    logger.info("Retrieved %d chunk(s) in total.", len(chunks))
    return chunks


@traced_tool("fetch_namespace_context", metadata={"pipeline": "retrieval", "stage": "retrieval"})
def fetch_namespace_context(
    namespaces: list[str],
    max_records_per_namespace: int | None = None,
) -> list[RetrievedChunk]:
    """
    Fetch namespace contents directly from Pinecone without query embedding.

    This mode is useful when the Pinecone namespace already represents the PDF
    scope the user wants to analyze, and the LLM should see all stored chunks
    from that namespace rather than nearest-neighbour search results.
    """
    if not namespaces:
        raise RetrievalError("At least one namespace must be specified for retrieval.")

    limit = (
        max_records_per_namespace
        if max_records_per_namespace is not None
        else settings.namespace_fetch_limit
    )
    logger.info(
        "Fetching namespace context directly | namespaces=%s limit_per_namespace=%s",
        namespaces,
        "all" if not limit else limit,
    )
    chunks = fetch_namespaces_chunks(
        namespaces=namespaces,
        max_records_per_namespace=limit,
    )
    logger.info("Fetched %d namespace context chunk(s).", len(chunks))
    return chunks
