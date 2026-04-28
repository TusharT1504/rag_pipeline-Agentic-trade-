"""
Singleton wrapper around a SentenceTransformer model.
Loading the model is expensive; this module ensures it happens once
per process and is reused across all embedding calls.
"""

import logging
import threading
from typing import Optional

from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from observability.langsmith import traced_function

logger = logging.getLogger(__name__)
settings = get_settings()

_model: Optional[SentenceTransformer] = None
_lock = threading.Lock()


@traced_function("get_embedding_model", metadata={"component": "embedding_model"})
def get_embedding_model() -> SentenceTransformer:
    """
    Return the process-level SentenceTransformer singleton.

    Thread-safe via a module-level lock so parallel FastAPI workers
    don't race to load the model.
    """
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:  # double-checked locking
            return _model
        model_name = settings.embedding_model
        logger.info("Loading SentenceTransformer model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info(
            "Model loaded | dimension=%d",
            _model.get_sentence_embedding_dimension(),
        )
    return _model


@traced_function("embed_texts", metadata={"component": "embedding_model"})
def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts into dense float vectors.

    Args:
        texts: Raw text strings to encode.
        batch_size: Number of texts processed per forward pass.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = get_embedding_model()
    logger.debug("Encoding %d text(s), batch_size=%d", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Encode a single query string.

    Args:
        query: The user's natural-language query.

    Returns:
        A single embedding vector.
    """
    return embed_texts([query])[0]
