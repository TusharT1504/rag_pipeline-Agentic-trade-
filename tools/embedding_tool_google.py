"""tools/embedding_tool_google.py

Embeds chunks using Google's free embedding model via the NEW google-genai SDK.

Model used : gemini-embedding-004  (replaces text-embedding-004)
Dimension  : 768
Free tier  : Yes
Batch limit: 100 texts per request (we use 50 to stay safe)

Install:  pip install google-genai
"""

from __future__ import annotations
import logging
import time
from typing import Iterator

from google import genai
from google.genai import types

from config import settings
from observability import compact_embedding_inputs, compact_embedding_outputs, traceable

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _client


def _batched(lst: list, size: int) -> Iterator[list]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


@traceable(
    run_type="embedding",
    name="Embed Chunks - Google",
    metadata={
        "provider": "google",
        "ls_model_name": settings.EMBEDDING_MODEL,
    },
    process_inputs=compact_embedding_inputs,
    process_outputs=compact_embedding_outputs,
)
def embedding_tool(chunks: list[dict], batch_size: int = 50) -> list[dict]:
    """
    Adds ``"embedding": list[float]`` to each chunk dict (in-place).

    Uses Google's gemini-embedding-004 (768-dim, free tier).

    Parameters
    ----------
    chunks : list[dict]   — output of chunking_tool
    batch_size : int      — texts per API call (max ~100)

    Returns
    -------
    list[dict]  — same dicts with embedding added
    """
    client = _get_client()
    model = settings.EMBEDDING_MODEL   # "gemini-embedding-004"
    total = len(chunks)

    logger.info("Embedding %d chunks with Google model '%s' …", total, model)

    for batch_num, batch in enumerate(_batched(chunks, batch_size), start=1):
        texts = [c["text"] for c in batch]

        while True:
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                # result.embeddings is a list of ContentEmbedding objects
                embeddings: list[list[float]] = [e.values for e in result.embeddings]
                break
            except Exception as exc:
                if "quota" in str(exc).lower() or "rate" in str(exc).lower():
                    logger.warning("Rate limited by Google — sleeping 30 s …")
                    time.sleep(30)
                else:
                    raise

        for chunk, emb in zip(batch, embeddings):
            chunk["embedding"] = emb

        logger.debug(
            "Batch %d/%d embedded (%d chunks)",
            batch_num,
            -(-total // batch_size),
            len(batch),
        )

    logger.info("Google embedding complete.")
    return chunks
