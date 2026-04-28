"""
Main Entrypoint.
Runs the FastAPI server via Uvicorn, or exposes the graph for direct use.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from config.settings import get_settings
from observability.langsmith import traced_function

settings = get_settings()


def configure_logging() -> None:
    """Set up structured logging for the entire application."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_server() -> None:
    """Start the Uvicorn server."""
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting RAG API on %s:%d (reload=%s)",
        settings.api_host,
        settings.api_port,
        settings.api_reload,
    )
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )


@traced_function("run_cli_query", metadata={"entrypoint": "cli", "mode": "query"})
def run_cli_query(query: str, namespaces: list[str] | None = None) -> str:
    """
    Run a single retrieval query from the CLI (no server required).

    Args:
        query: Natural-language question.
        namespaces: Pinecone namespaces to search.

    Returns:
        The generated answer string.
    """
    configure_logging()
    from graph.rag_graph import get_rag_graph
    from graph.state import RAGState, get_state_value

    logger = logging.getLogger(__name__)
    logger.info("CLI query: %s", query)

    state = RAGState(
        ingest_flag=False,
        query=query,
        namespaces=namespaces or settings.default_namespaces,
        top_k=settings.retrieval_top_k,
    )
    graph = get_rag_graph()
    result = graph.invoke(state)

    error = get_state_value(result, "error", "")
    if error:
        logger.error("Query failed: %s", error)
    return get_state_value(result, "answer", "")


@traced_function("run_cli_ingest", metadata={"entrypoint": "cli", "mode": "ingest"})
def run_cli_ingest(pdf_paths: list[str], sector: str) -> int:
    """
    Run the ingestion pipeline from the CLI.

    Args:
        pdf_paths: Paths to PDF files.
        sector: Pinecone namespace / sector label.

    Returns:
        Number of vectors upserted.
    """
    configure_logging()
    from graph.rag_graph import get_rag_graph
    from graph.state import RAGState, get_state_value

    logger = logging.getLogger(__name__)
    logger.info("CLI ingest: sector=%s files=%s", sector, pdf_paths)

    state = RAGState(
        ingest_flag=True,
        pdf_paths=pdf_paths,
        sector=sector,
    )
    graph = get_rag_graph()
    result = graph.invoke(state)

    error = get_state_value(result, "error", "")
    upserted_count = get_state_value(result, "upserted_count", 0)
    if error:
        logger.error("Ingestion failed: %s", error)
    else:
        logger.info("Ingested %d vectors into namespace '%s'.", upserted_count, sector)
    return upserted_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG System CLI")
    sub = parser.add_subparsers(dest="command")

    # serve sub-command
    sub.add_parser("serve", help="Start the FastAPI server")

    # query sub-command
    q_parser = sub.add_parser("query", help="Run a single retrieval query")
    q_parser.add_argument("query", type=str, help="Question to answer")
    q_parser.add_argument(
        "--namespaces", nargs="+", default=None, help="Pinecone namespaces"
    )

    # ingest sub-command
    i_parser = sub.add_parser("ingest", help="Ingest PDFs into Pinecone")
    i_parser.add_argument("pdfs", nargs="+", help="PDF file paths")
    i_parser.add_argument("--sector", required=True, help="Sector / namespace")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        run_server()
    elif args.command == "query":
        answer = run_cli_query(args.query, args.namespaces)
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
    elif args.command == "ingest":
        count = run_cli_ingest(args.pdfs, args.sector)
        print(f"\nUpserted {count} vectors to namespace '{args.sector}'.")
