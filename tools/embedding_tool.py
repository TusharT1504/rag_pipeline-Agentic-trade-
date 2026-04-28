"""
Embedding Tool.
Used for:
  - Query embedding (ALWAYS, in retrieval mode)
  - Document embedding (ONLY in ingestion mode)

This module NEVER writes to Pinecone — that responsibility belongs
exclusively to vector_store_tool.py.
"""

from __future__ import annotations

import logging

from graph.state import DocumentChunk
from observability.langsmith import traced_tool
from tools.st_model import embed_query as _embed_query, embed_texts

logger = logging.getLogger(__name__)


@traced_tool("embed_query", metadata={"pipeline": "retrieval", "stage": "embedding"})
def embed_query(query: str) -> list[float]:
    """
    Produce a single embedding vector for a user query.

    Args:
        query: Natural-language query string.

    Returns:
        Dense float vector of dimension = embedding model dimension.
    """
    logger.debug("Embedding query (len=%d)", len(query))
    vector = _embed_query(query)
    logger.debug("Query embedded | vector_dim=%d", len(vector))
    return vector


@traced_tool("embed_document_chunks", metadata={"pipeline": "ingestion", "stage": "embedding"})
def embed_document_chunks(
    chunks: list[DocumentChunk],
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Embed a list of ``DocumentChunk`` objects.

    This function is ONLY called during ingestion mode.
    Returns a list of vectors in the same order as *chunks*.

    Args:
        chunks: Chunks produced by the chunking stage.
        batch_size: SentenceTransformer encoding batch size.

    Returns:
        List of embedding vectors (parallel to *chunks*).
    """
    if not chunks:
        logger.warning("embed_document_chunks called with empty chunk list.")
        return []

    texts = [chunk.text for chunk in chunks]
    logger.info("Embedding %d document chunks (batch_size=%d)…", len(texts), batch_size)
    vectors = embed_texts(texts, batch_size=batch_size)
    logger.info("Document chunks embedded | count=%d", len(vectors))
    return vectors
