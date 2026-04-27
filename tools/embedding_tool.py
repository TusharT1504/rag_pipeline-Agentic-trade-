"""tools/embedding_tool.py

Single entry-point for all embedding providers.
Controlled by EMBEDDING_PROVIDER in your .env:

    EMBEDDING_PROVIDER=google                → Google gemini-embedding-004 (free, 768-dim)
    EMBEDDING_PROVIDER=sentence_transformers → Local SentenceTransformer (free, no API)
"""

from __future__ import annotations
import logging
import time
from typing import Iterator

from config import settings
from observability import compact_embedding_inputs, compact_embedding_outputs, traceable
from tools.st_model import get_model

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _batched(lst: list, size: int) -> Iterator[list]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ── Provider implementations ─────────────────────────────────────────────────

def _embed_google(chunks: list[dict], batch_size: int = 50) -> list[dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    model  = settings.EMBEDDING_MODEL   # e.g. "gemini-embedding-004"
    total  = len(chunks)
    logger.info("Google embedding: %d chunks with '%s' ...", total, model)

    for batch_num, batch in enumerate(_batched(chunks, batch_size), start=1):
        texts = [c["text"] for c in batch]
        while True:
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                embeddings = [e.values for e in result.embeddings]
                break
            except Exception as exc:
                if "quota" in str(exc).lower() or "rate" in str(exc).lower():
                    logger.warning("Google rate limit - sleeping 30 s ...")
                    time.sleep(30)
                else:
                    raise
        for chunk, emb in zip(batch, embeddings):
            chunk["embedding"] = emb
        logger.debug("Batch %d/%d done.", batch_num, -(-total // batch_size))

    logger.info("Google embedding complete.")
    return chunks


def _embed_sentence_transformers(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    model = get_model()
    total = len(chunks)
    logger.info("ST embedding: %d chunks ...", total)

    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=total > 50,
        convert_to_numpy=True,
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    logger.info("SentenceTransformer embedding complete.")
    return chunks


# ── Public entry point ────────────────────────────────────────────────────────

@traceable(
    run_type="embedding",
    name="Embed Chunks",
    metadata={
        "provider": settings.EMBEDDING_PROVIDER,
        "ls_provider": settings.EMBEDDING_PROVIDER,
        "ls_model_name": settings.EMBEDDING_MODEL,
    },
    process_inputs=compact_embedding_inputs,
    process_outputs=compact_embedding_outputs,
)
def embedding_tool(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks using the provider set in EMBEDDING_PROVIDER.

    Parameters
    ----------
    chunks : list[dict]   - output of chunking_tool
                            each item must have a "text" key

    Returns
    -------
    list[dict]  - same dicts with "embedding": list[float] added in-place
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "google":
        return _embed_google(chunks)
    elif provider == "sentence_transformers":
        return _embed_sentence_transformers(chunks)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{provider}'. "
            "Choose from: google | sentence_transformers"
        )
