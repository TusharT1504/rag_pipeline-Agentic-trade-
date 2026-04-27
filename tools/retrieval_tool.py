"""tools/retrieval_tool.py

Single entry-point for query embedding + retrieval.
Uses the same EMBEDDING_PROVIDER as embedding_tool.py so vectors always match.

    EMBEDDING_PROVIDER=google                → Google gemini-embedding-004
    EMBEDDING_PROVIDER=sentence_transformers → Local SentenceTransformer
"""

from __future__ import annotations
import logging

from config import settings
from observability import retrieved_chunks_as_documents, traceable
from tools.st_model import get_model
from tools.vector_store_tool import similarity_search

logger = logging.getLogger(__name__)

# ── Provider implementations ─────────────────────────────────────────────────

def _query_embed_google(query: str) -> list[float]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    result = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def _query_embed_sentence_transformers(query: str) -> list[float]:
    model = get_model()
    return model.encode(query, convert_to_numpy=True).tolist()


# ── Public entry point ────────────────────────────────────────────────────────

@traceable(
    run_type="retriever",
    name="Retrieve Chunks",
    metadata={
        "provider": settings.EMBEDDING_PROVIDER,
        "ls_provider": settings.EMBEDDING_PROVIDER,
        "ls_model_name": settings.EMBEDDING_MODEL,
    },
    process_outputs=retrieved_chunks_as_documents,
)
def retrieval_tool(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed the query using the configured provider, then fetch top-k chunks
    from Pinecone.

    Parameters
    ----------
    query  : str
    top_k  : int | None  — overrides settings.TOP_K if provided

    Returns
    -------
    list[dict]  — [{"text": str, "score": float, "metadata": dict}, ...]
                  sorted by descending relevance score
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    logger.info("Embedding query with provider '%s': '%s'", provider, query[:80])

    if provider == "google":
        query_embedding = _query_embed_google(query)
    elif provider == "sentence_transformers":
        query_embedding = _query_embed_sentence_transformers(query)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{provider}'. "
            "Choose from: google | sentence_transformers"
        )

    chunks = similarity_search(query_embedding, top_k=top_k)

    # Lightweight keyword re-ranking boost
    keywords = set(query.lower().split())
    for chunk in chunks:
        overlap = sum(1 for kw in keywords if kw in chunk["text"].lower())
        chunk["score"] = round(chunk["score"] + overlap * 0.005, 4)
        chunk["page_content"] = chunk["text"]
        chunk["type"] = "Document"

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks
