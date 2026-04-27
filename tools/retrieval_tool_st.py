"""tools/retrieval_tool_st.py

Query-embedding variant that uses the same local SentenceTransformer model
that was used during indexing, then fetches top-k chunks from Pinecone.
"""

from __future__ import annotations
import logging

from config import settings
from observability import retrieved_chunks_as_documents, traceable
from tools.st_model import get_model
from tools.vector_store_tool import similarity_search

logger = logging.getLogger(__name__)


@traceable(
    run_type="retriever",
    name="Retrieve Chunks - SentenceTransformers",
    metadata={
        "provider": "sentence_transformers",
        "ls_provider": "sentence_transformers",
        "ls_model_name": settings.EMBEDDING_MODEL,
    },
    process_outputs=retrieved_chunks_as_documents,
)
def retrieval_tool(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed *query* with the local SentenceTransformer model and retrieve top-k chunks.

    IMPORTANT: must use the same model that was used in embedding_tool_st.py

    Parameters
    ----------
    query : str
    top_k : int | None

    Returns
    -------
    list[dict]  —  [{"text": str, "score": float, "metadata": dict}, ...]
    """
    model = get_model()

    logger.info("ST-embedding query: '%s'", query[:80])
    query_embedding: list[float] = model.encode(query, convert_to_numpy=True).tolist()

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
