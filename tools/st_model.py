"""tools/st_model.py

Shared SentenceTransformer loader so embedding and retrieval reuse
the same in-process model instance.
"""

from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

_model = None


def get_model():
    """Load the SentenceTransformer once per process.

    Optimization:
    - Try local cache first to avoid repeated HF network HEAD checks.
    - If cache is missing, download once and reuse.
    """
    global _model
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    model_name = settings.EMBEDDING_MODEL
    logger.info("Initializing SentenceTransformer '%s' ...", model_name)

    try:
        _model = SentenceTransformer(model_name, local_files_only=True)
        logger.info("Loaded SentenceTransformer '%s' from local cache.", model_name)
    except Exception:
        logger.info(
            "Local cache miss for '%s'. Downloading from Hugging Face once ...",
            model_name,
        )
        _model = SentenceTransformer(model_name)
        logger.info("Downloaded and loaded SentenceTransformer '%s'.", model_name)

    return _model
