"""
PDF Loader Tool — Ingestion pipeline, Stage 1.
Loads PDF files and returns page-level dictionaries that feed into
the chunking stage. Only used in ingestion mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pypdf

from observability.langsmith import traced_tool

logger = logging.getLogger(__name__)


class PDFLoaderError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


@traced_tool("load_pdf", metadata={"pipeline": "ingestion", "stage": "pdf_loader"})
def load_pdf(pdf_path: str, sector: str) -> list[dict[str, Any]]:
    """
    Load a single PDF file and return a list of page-level dicts.

    Each dict contains:
    - ``text``: extracted text for the page
    - ``page_number``: 1-based page index
    - ``source_file``: basename of the PDF
    - ``sector``: sector / namespace label

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        sector: Sector label used as the Pinecone namespace.

    Returns:
        List of page dicts (one per page).

    Raises:
        PDFLoaderError: If the file does not exist or cannot be parsed.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise PDFLoaderError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise PDFLoaderError(f"Expected a .pdf file, got: {pdf_path}")

    logger.info("Loading PDF: %s", pdf_path)
    pages: list[dict[str, Any]] = []

    try:
        reader = pypdf.PdfReader(str(path))
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                logger.debug("Page %d is empty, skipping.", idx)
                continue
            pages.append(
                {
                    "text": text,
                    "page_number": idx,
                    "source_file": path.name,
                    "sector": sector,
                }
            )
    except Exception as exc:
        raise PDFLoaderError(f"Failed to parse {pdf_path}: {exc}") from exc

    logger.info("Loaded %d non-empty pages from %s", len(pages), path.name)
    return pages


@traced_tool("load_pdfs", metadata={"pipeline": "ingestion", "stage": "pdf_loader"})
def load_pdfs(pdf_paths: list[str], sector: str) -> list[dict[str, Any]]:
    """
    Load multiple PDF files for the same sector.

    Args:
        pdf_paths: List of paths to PDF files.
        sector: Shared sector label for all files.

    Returns:
        Aggregated list of page dicts across all files.
    """
    all_pages: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        try:
            pages = load_pdf(pdf_path, sector)
            all_pages.extend(pages)
        except PDFLoaderError as exc:
            logger.error("Skipping file due to error: %s", exc)
    logger.info("Total pages loaded across %d file(s): %d", len(pdf_paths), len(all_pages))
    return all_pages
