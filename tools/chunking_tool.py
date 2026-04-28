"""
Chunking Tool — Ingestion pipeline, Stage 2.
Splits page-level text into overlapping chunks suitable for embedding.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from config.settings import get_settings
from graph.state import DocumentChunk
from observability.langsmith import traced_tool

logger = logging.getLogger(__name__)
settings = get_settings()


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split *text* into overlapping windows of *chunk_size* characters.

    Args:
        text: Raw text to split.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap  # slide with overlap
    return chunks


@traced_tool("chunk_pages", metadata={"pipeline": "ingestion", "stage": "chunking"})
def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Convert page-level dicts into ``DocumentChunk`` objects.

    Args:
        pages: Output of ``pdf_loader_tool.load_pdfs``.
        chunk_size: Override settings value if provided.
        chunk_overlap: Override settings value if provided.

    Returns:
        Flat list of ``DocumentChunk`` objects ready for embedding.
    """
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap

    chunks: list[DocumentChunk] = []
    for page in pages:
        text = page["text"]
        sub_chunks = _split_text(text, cs, co)
        for sub_idx, sub_text in enumerate(sub_chunks):
            chunk_id = str(uuid.uuid4())
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=sub_text,
                    sector=page["sector"],
                    source_file=page["source_file"],
                    page_number=page["page_number"],
                    metadata={
                        "sub_chunk_index": sub_idx,
                        "char_length": len(sub_text),
                    },
                )
            )

    logger.info(
        "Created %d chunks from %d page(s) | chunk_size=%d overlap=%d",
        len(chunks),
        len(pages),
        cs,
        co,
    )
    return chunks
