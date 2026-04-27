"""tools/embedding_tool_st.py

Embeds chunks using a local Sentence-Transformers model.
Runs 100% offline — no API key, no cost.

Recommended models
------------------
Model name                              Dim   Notes
--------------------------------------  ----  ---------------------------------
all-MiniLM-L6-v2                        384   Fast, good quality, ~80 MB
all-mpnet-base-v2                       768   Better quality, ~420 MB
multi-qa-MiniLM-L6-cos-v1              384   Tuned for QA / retrieval
BAAI/bge-small-en-v1.5                  384   Strong retrieval benchmark score

Set ST_MODEL in .env (or leave blank for the default below).
"""

from __future__ import annotations
import logging

from tools.st_model import get_model
from config import settings
from observability import compact_embedding_inputs, compact_embedding_outputs, traceable

logger = logging.getLogger(__name__)


@traceable(
    run_type="embedding",
    name="Embed Chunks - SentenceTransformers",
    metadata={
        "provider": "sentence_transformers",
        "ls_provider": "sentence_transformers",
        "ls_model_name": settings.EMBEDDING_MODEL,
    },
    process_inputs=compact_embedding_inputs,
    process_outputs=compact_embedding_outputs,
)
def embedding_tool(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """
    Adds ``"embedding": list[float]`` to each chunk dict (in-place).

    Uses a local Sentence-Transformers model — no internet required after
    first download.

    Parameters
    ----------
    chunks : list[dict]   — output of chunking_tool
    batch_size : int      — texts per encode() call

    Returns
    -------
    list[dict]  — same dicts with embedding added
    """
    model = get_model()
    total = len(chunks)
    logger.info("Embedding %d chunks locally with SentenceTransformer …", total)

    texts = [c["text"] for c in chunks]

    # encode() accepts a list and handles batching internally;
    # show_progress_bar gives visual feedback for large corpuses
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=total > 50,
        convert_to_numpy=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()   # numpy → plain list for Pinecone

    logger.info("SentenceTransformer embedding complete.")
    return chunks
