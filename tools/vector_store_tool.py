"""tools/vector_store_tool.py

Handles all Pinecone operations:
  • index_exists()          — check whether vectors are already stored
  • upsert_chunks()         — store embeddings + metadata
  • similarity_search()     — retrieve top-k chunks for a query embedding
"""

from __future__ import annotations
import logging
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config import settings
from observability import (
    compact_embedding_inputs,
    compact_search_inputs,
    compact_search_outputs,
    traceable,
)

logger = logging.getLogger(__name__)

_pc: Pinecone | None = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def _get_index_dimension(index_description: Any) -> int | None:
    """Extract the dimension from Pinecone's dict-like or object response."""
    if isinstance(index_description, dict):
        dimension = index_description.get("dimension")
    else:
        dimension = getattr(index_description, "dimension", None)
    return int(dimension) if dimension is not None else None


def _validate_index_dimension(pc: Pinecone, index_name: str) -> None:
    """Fail early if an existing index was created for another embedding size."""
    description = pc.describe_index(index_name)
    index_dimension = _get_index_dimension(description)

    if index_dimension is None:
        logger.warning("Could not verify dimension for Pinecone index '%s'.", index_name)
        return

    if index_dimension != settings.EMBEDDING_DIMENSION:
        raise ValueError(
            f"Pinecone index '{index_name}' has dimension {index_dimension}, "
            f"but EMBEDDING_DIMENSION is {settings.EMBEDDING_DIMENSION}. "
            "Use a Pinecone index created with the same dimension as your "
            "embedding model, or update PINECONE_INDEX_NAME in .env."
        )


def _get_or_create_index():
    """Return the Pinecone Index object, creating it on first use."""
    pc = _get_client()
    index_name = settings.PINECONE_INDEX_NAME

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' …", index_name)
        pc.create_index(
            name=index_name,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.PINECONE_ENVIRONMENT,
            ),
        )
        logger.info("Index '%s' created.", index_name)
    else:
        _validate_index_dimension(pc, index_name)
        logger.info("Using existing Pinecone index '%s'.", index_name)

    return pc.Index(index_name)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

@traceable(
    run_type="tool",
    name="Pinecone Index Check",
    metadata={"index_name": settings.PINECONE_INDEX_NAME},
)
def index_exists() -> bool:
    """Return True if the index exists AND has at least one vector."""
    pc = _get_client()
    index_name = settings.PINECONE_INDEX_NAME

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        return False

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total_vectors: int = stats.get("total_vector_count", 0)
    logger.info("Pinecone index '%s' has %d vectors.", index_name, total_vectors)
    return total_vectors > 0


@traceable(
    run_type="tool",
    name="Pinecone Upsert Chunks",
    metadata={"index_name": settings.PINECONE_INDEX_NAME},
    process_inputs=compact_embedding_inputs,
)
def upsert_chunks(chunks: list[dict], batch_size: int = 100) -> int:
    """
    Upsert embeddings + metadata into Pinecone.

    Parameters
    ----------
    chunks : list[dict]
        Must have keys: ``id``, ``embedding``, ``text``, ``metadata``.

    Returns
    -------
    int  — number of vectors upserted
    """
    index = _get_or_create_index()
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors: list[dict[str, Any]] = []

        for chunk in batch:
            embedding = chunk["embedding"]
            if len(embedding) != settings.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Chunk '{chunk['id']}' has embedding dimension {len(embedding)}, "
                    f"but EMBEDDING_DIMENSION is {settings.EMBEDDING_DIMENSION}. "
                    "Check EMBEDDING_MODEL and EMBEDDING_DIMENSION in .env."
                )

            meta = dict(chunk["metadata"])
            meta["chunk_text"] = chunk["text"]   # store raw text in metadata
            vectors.append(
                {
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": meta,
                }
            )

        index.upsert(vectors=vectors)
        total_upserted += len(vectors)
        logger.debug("Upserted batch of %d vectors.", len(vectors))

    logger.info("Total vectors upserted: %d", total_upserted)
    return total_upserted


@traceable(
    run_type="tool",
    name="Pinecone Similarity Search",
    metadata={"index_name": settings.PINECONE_INDEX_NAME},
    process_inputs=compact_search_inputs,
    process_outputs=compact_search_outputs,
)
def similarity_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    """
    Perform a cosine similarity search.

    Returns
    -------
    list[dict]
        [{"text": str, "score": float, "metadata": dict}, ...]
        sorted by descending score.
    """
    index = _get_or_create_index()
    k = top_k or settings.TOP_K

    if len(query_embedding) != settings.EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query embedding has dimension {len(query_embedding)}, "
            f"but EMBEDDING_DIMENSION is {settings.EMBEDDING_DIMENSION}. "
            "Check EMBEDDING_MODEL and EMBEDDING_DIMENSION in .env."
        )

    response = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True,
    )

    results = []
    for match in response.get("matches", []):
        meta = dict(match.get("metadata", {}))
        text = meta.pop("chunk_text", "")
        results.append(
            {
                "text": text,
                "score": round(float(match.get("score", 0.0)), 4),
                "metadata": meta,
            }
        )

    logger.info("Retrieved %d chunks (top_k=%d).", len(results), k)
    return results
