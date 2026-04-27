"""tools/pdf_loader_tool.py

Loads every PDF from the configured directory and returns a list of
{document_name, page_number, text} dicts — one entry per page.
"""

from __future__ import annotations
import logging
from pathlib import Path

from pypdf import PdfReader

from config import settings
from observability import compact_pages_outputs, traceable

logger = logging.getLogger(__name__)


@traceable(
    run_type="tool",
    name="Load PDFs",
    process_outputs=compact_pages_outputs,
)
def pdf_loader_tool() -> list[dict]:
    """
    Scans PDF_DIR for *.pdf files, extracts text page-by-page.

    Returns
    -------
    list[dict]  —  [{"document_name": str, "page_number": int, "text": str}, ...]
    """
    pdf_dir = Path(settings.PDF_DIR)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir.resolve()}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir.resolve()}")

    pages: list[dict] = []

    for pdf_path in pdf_files:
        logger.info("Loading PDF: %s", pdf_path.name)
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue  # skip empty/scanned pages
                pages.append(
                    {
                        "document_name": pdf_path.name,
                        "page_number": page_num,
                        "text": text,
                    }
                )
        except Exception as exc:
            logger.error("Failed to load %s: %s", pdf_path.name, exc)
            raise

    logger.info("Loaded %d pages from %d PDFs", len(pages), len(pdf_files))
    return pages
