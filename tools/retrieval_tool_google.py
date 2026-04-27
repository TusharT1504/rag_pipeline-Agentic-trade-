"""tools/retrieval_tool_google.py

Query-embedding variant using the NEW google-genai SDK (google.genai).
Uses task_type RETRIEVAL_QUERY for query vectors (different from RETRIEVAL_DOCUMENT).
"""

from __future__ import annotations
import logging

from google import genai
from google.genai import types

from config import settings
from observability import retrieved_chunks_as_documents, traceable
from tools.vector_store_tool import similarity_search

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _client


@traceable(
    run_type="retriever",
    name="Retrieve Chunks - Google",
    metadata={
        "provider": "google",
        "ls_provider": "google_genai",
        "ls_model_name": settings.EMBEDDING_MODEL,
    },
    process_outputs=retrieved_chunks_as_documents,
)
def retrieval_tool(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed *query* with Google gemini-embedding-004 and retrieve top-k chunks.

    Parameters
    ----------
    query : str
    top_k : int | None

    Returns
    -------
    list[dict]  —  [{"text": str, "score": float, "metadata": dict}, ...]
    """
    client = _get_client()
    model = settings.EMBEDDING_MODEL   # "gemini-embedding-004"

    logger.info("Google-embedding query: '%s'", query[:80])
    result = client.models.embed_content(
        model=model,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",   # ← must differ from document task_type
        ),
    )
    query_embedding: list[float] = result.embeddings[0].values

    chunks = similarity_search(query_embedding, top_k=top_k)

    # lightweight keyword re-ranking
    keywords = set(query.lower().split())
    for chunk in chunks:
        overlap = sum(1 for kw in keywords if kw in chunk["text"].lower())
        chunk["score"] = round(chunk["score"] + overlap * 0.005, 4)
        chunk["page_content"] = chunk["text"]
        chunk["type"] = "Document"

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks
