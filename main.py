"""main.py — Command-line entry point for the LangGraph RAG system.

Usage
-----
    python main.py "What was the export growth in 2023?"
    python main.py --top-k 8 "Summarise the key findings in the report."
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
import os
import sys


def _use_project_venv_for_cli() -> None:
    """Relaunch the CLI with the project venv when run from system Python."""
    if __name__ != "__main__" or sys.prefix != sys.base_prefix:
        return

    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if current_python != target_python:
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


_use_project_venv_for_cli()

from config import settings
from graph import RAGState, rag_graph
from observability import traceable


def _configure_third_party_logging() -> None:
    """Reduce noisy library logs while keeping app-level logs visible."""
    if not settings.QUIET_THIRD_PARTY_LOGS:
        return

    # Hide verbose HF/HTTP internals and progress output.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    noisy_loggers = (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "sentence_transformers.base.model",
        "transformers",
    )
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        # transformers may not be installed for non-ST setups.
        pass


log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
_configure_third_party_logging()
logger = logging.getLogger(__name__)


def _compact_run_query_outputs(output: dict) -> dict:
    return {
        "answer_preview": output.get("answer", "")[:300],
        "source_count": len(output.get("sources", [])),
        "confidence": output.get("confidence"),
        "tool_analysis": output.get("tool_analysis", {}),
    }


@traceable(
    run_type="chain",
    name="RAG Query",
    metadata={"service": "rag-pipeline"},
    process_outputs=_compact_run_query_outputs,
)
def run_query(query: str, top_k: int | None = None) -> dict:
    """
    Execute the full RAG graph for a given query.

    Returns
    -------
    dict  —  {"answer": ..., "sources": [...], "confidence": ...}
    """
    settings.validate()

    initial_state = RAGState(query=query)
    if top_k:
        settings.TOP_K = top_k

    logger.info("Starting RAG graph for query: '%s'", query)
    raw = rag_graph.invoke(initial_state)

    # LangGraph may return a RAGState object OR a plain dict depending on version
    if isinstance(raw, dict):
        error        = raw.get("error")
        final_answer = raw.get("final_answer")
        tool_analysis = raw.get("tool_analysis", {})
    else:
        error        = raw.error
        final_answer = raw.final_answer
        tool_analysis = raw.tool_analysis

    default_result = {
        "answer": "The information is not available in the provided documents.",
        "sources": [],
        "confidence": "low",
    }

    if error:
        logger.error("Graph error: %s", error)
        result = default_result.copy()
        result["answer"] = f"An error occurred: {error}"
    else:
        result = final_answer or default_result
    result["tool_analysis"] = tool_analysis
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph RAG system - query your PDFs")
    parser.add_argument("query", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of chunks to retrieve (overrides settings)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    args = parser.parse_args()

    answer = run_query(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(answer, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 60)
        print(f"ANSWER:\n{answer['answer']}")
        print(f"\nCONFIDENCE: {answer['confidence'].upper()}")
        if answer.get("tool_analysis"):
            analysis = answer["tool_analysis"]
            scores = analysis.get("score_summary", {})
            print(
                "\nTOOL ANALYSIS: "
                f"{analysis.get('retrieved_count', 0)} chunks, "
                f"top score={scores.get('top')}, "
                f"sources={len(analysis.get('source_coverage', {}))}"
            )
        if answer["sources"]:
            print("\nSOURCES:")
            for s in answer["sources"]:
                print(f"  - {s.get('document_name')}  p.{s.get('page_number')}  [{s.get('section')}]")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
