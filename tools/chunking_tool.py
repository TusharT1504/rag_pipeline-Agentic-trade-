"""tools/chunking_tool.py

Splits raw page-text into overlapping semantic chunks and attaches rich
metadata so every chunk is independently traceable back to its source.
"""

from __future__ import annotations
import re
import logging
from datetime import datetime, timezone

from config import settings
from observability import (
    compact_embedding_outputs,
    compact_pages_inputs,
    traceable,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic section detector
# ---------------------------------------------------------------------------
_SECTION_PATTERNS = [
    re.compile(r"^\s{0,4}(\d+[\.\d]*)\s+([A-Z][^\n]{3,60})\s*$", re.MULTILINE),
    re.compile(r"^\s{0,4}([A-Z][A-Z\s]{4,50})\s*$", re.MULTILINE),
]


def _detect_section(text: str) -> str:
    """Return the first plausible section heading found in *text*, or 'General'."""
    for pattern in _SECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0).strip()[:80]
    return "General"


# ---------------------------------------------------------------------------
# Sliding-window chunker (token-aware via character proxy)
# ---------------------------------------------------------------------------

@traceable(
    run_type="tool",
    name="Chunk Documents",
    process_inputs=compact_pages_inputs,
    process_outputs=compact_embedding_outputs,
)
def chunking_tool(pages: list[dict]) -> list[dict]:
    """
    Parameters
    ----------
    pages : list[dict]
        Output of pdf_loader_tool — each item has
        ``document_name``, ``page_number``, ``text``.

    Returns
    -------
    list[dict]
        Each item:
        {
            "id":       "<doc_stem>_page<N>_chunk<M>",
            "text":     str,
            "metadata": {
                "document_name", "page_number", "section",
                "chunk_index", "source", "created_at"
            }
        }
    """
    chunk_size = settings.CHUNK_SIZE        # chars (approx 1 token ≈ 4 chars)
    chunk_overlap = settings.CHUNK_OVERLAP

    chunks: list[dict] = []
    created_at = datetime.now(timezone.utc).isoformat()

    for page in pages:
        doc_name: str = page["document_name"]
        page_num: int = page["page_number"]
        text: str = page["text"]

        # derive a short source tag from the filename
        source_tag = doc_name.rsplit(".", 1)[0]   # strip ".pdf"
        section = _detect_section(text)

        # sliding window over the page text
        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{source_tag}_page{page_num}_chunk{chunk_index}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "document_name": doc_name,
                            "page_number": page_num,
                            "section": section,
                            "chunk_index": chunk_index,
                            "source": source_tag,
                            "created_at": created_at,
                        },
                    }
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = end - chunk_overlap   # overlap for context continuity

    logger.info(
        "Chunking complete: %d chunks from %d pages", len(chunks), len(pages)
    )
    return chunks
