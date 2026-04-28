"""
LangGraph Node Definitions.

All nodes:
  - Accept and return RAGState
  - Are decorated with @traced_node for LangSmith observability
  - Log structured metadata after each stage
  - Handle errors gracefully by setting state.error

Ingestion nodes (only run when ingest_flag == True):
  1. pdf_loader_node
  2. chunking_node
  3. embedding_node  (document side)
  4. vector_store_node

Retrieval nodes (default path):
  1. retrieval_node  (direct namespace fetch)
  2. answer_generation_node
"""

from __future__ import annotations

import logging

from graph.state import RAGState
from observability.langsmith import traced_node, log_node_metadata
from tools.pdf_loader_tool import load_pdfs
from tools.chunking_tool import chunk_pages
from tools.embedding_tool import embed_document_chunks
from tools.vector_store_tool import upsert_chunks
from tools.retrieval_tool import fetch_namespace_context
from tools.answer_generation_tool import generate_answer
from tools.tool_analysis import (
    analyse_chunks,
    analyse_embeddings,
    analyse_retrieved_chunks,
    format_context_for_display,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION NODES
# ═══════════════════════════════════════════════════════════════════════════════


@traced_node("pdf_loader_node", metadata={"pipeline": "ingestion", "stage": 1})
def pdf_loader_node(state: RAGState) -> RAGState:
    """
    Node 1 (Ingestion): Load PDFs from disk into page-level dicts.

    Reads state.pdf_paths and state.sector.
    Writes state.raw_documents.
    """
    logger.info("[pdf_loader_node] Loading %d PDF(s) for sector='%s'", len(state.pdf_paths), state.sector)

    if not state.pdf_paths:
        state.error = "pdf_loader_node: No PDF paths provided."
        logger.error(state.error)
        return state

    if not state.sector:
        state.error = "pdf_loader_node: sector must be set before loading PDFs."
        logger.error(state.error)
        return state

    try:
        raw_docs = load_pdfs(state.pdf_paths, state.sector)
        state.raw_documents = raw_docs
        log_node_metadata(
            "pdf_loader_node",
            extra={
                "pdf_count": len(state.pdf_paths),
                "pages_loaded": len(raw_docs),
                "sector": state.sector,
            },
        )
    except Exception as exc:
        state.error = f"pdf_loader_node failed: {exc}"
        logger.error(state.error)

    return state


@traced_node("chunking_node", metadata={"pipeline": "ingestion", "stage": 2})
def chunking_node(state: RAGState) -> RAGState:
    """
    Node 2 (Ingestion): Split raw pages into overlapping text chunks.

    Reads state.raw_documents.
    Writes state.document_chunks.
    """
    if state.error:
        logger.warning("[chunking_node] Skipping due to upstream error: %s", state.error)
        return state

    logger.info("[chunking_node] Chunking %d page(s)…", len(state.raw_documents))

    try:
        chunks = chunk_pages(state.raw_documents)
        state.document_chunks = chunks
        analysis = analyse_chunks(chunks)
        log_node_metadata(
            "chunking_node",
            extra={"chunk_analysis": analysis},
        )
    except Exception as exc:
        state.error = f"chunking_node failed: {exc}"
        logger.error(state.error)

    return state


@traced_node("embedding_node", metadata={"pipeline": "ingestion", "stage": 3})
def embedding_node(state: RAGState) -> RAGState:
    """
    Node 3 (Ingestion): Embed document chunks.

    Reads state.document_chunks.
    Writes state.chunk_embeddings.
    """
    if state.error:
        logger.warning("[embedding_node] Skipping due to upstream error: %s", state.error)
        return state

    logger.info("[embedding_node] Embedding %d chunk(s)…", len(state.document_chunks))

    try:
        vectors = embed_document_chunks(state.document_chunks)
        state.chunk_embeddings = vectors
        analysis = analyse_embeddings(vectors)
        log_node_metadata(
            "embedding_node",
            extra={"embedding_analysis": analysis},
        )
    except Exception as exc:
        state.error = f"embedding_node failed: {exc}"
        logger.error(state.error)

    return state


@traced_node("vector_store_node", metadata={"pipeline": "ingestion", "stage": 4})
def vector_store_node(state: RAGState) -> RAGState:
    """
    Node 4 (Ingestion): Upsert chunk embeddings into Pinecone.

    Reads state.document_chunks and state.chunk_embeddings.
    Writes state.upserted_count.
    """
    if state.error:
        logger.warning("[vector_store_node] Skipping due to upstream error: %s", state.error)
        return state

    logger.info(
        "[vector_store_node] Upserting %d vector(s) to Pinecone…",
        len(state.chunk_embeddings),
    )

    try:
        count = upsert_chunks(state.document_chunks, state.chunk_embeddings)
        state.upserted_count = count
        log_node_metadata(
            "vector_store_node",
            namespace=state.sector,
            extra={"upserted_count": count},
        )
    except Exception as exc:
        state.error = f"vector_store_node failed: {exc}"
        logger.error(state.error)

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL NODES
# ═══════════════════════════════════════════════════════════════════════════════


@traced_node("retrieval_node", metadata={"pipeline": "retrieval", "stage": 2})
def retrieval_node(state: RAGState) -> RAGState:
    """
    Node 1 (Retrieval): Fetch namespace context directly from Pinecone.

    Reads state.namespaces.
    Writes state.retrieved_chunks.
    """
    if state.error:
        logger.warning("[retrieval_node] Skipping due to upstream error: %s", state.error)
        return state

    namespaces = state.namespaces or settings.default_namespaces
    logger.info(
        "[retrieval_node] Fetching all context from namespace(s)=%s…",
        namespaces,
    )

    try:
        chunks = fetch_namespace_context(namespaces=namespaces)
        state.retrieved_chunks = chunks
        analysis = analyse_retrieved_chunks(chunks)
        log_node_metadata(
            "retrieval_node",
            query=state.query,
            namespace=namespaces,
            retrieved_docs_count=len(chunks),
            extra={"retrieval_analysis": analysis},
        )
        logger.debug(format_context_for_display(chunks))
    except Exception as exc:
        state.error = f"retrieval_node failed: {exc}"
        logger.error(state.error)

    return state


@traced_node("answer_generation_node", metadata={"pipeline": "retrieval", "stage": 3})
def answer_generation_node(state: RAGState) -> RAGState:
    """
    Node 3 (Retrieval): Generate a grounded answer using the Groq LLM.

    Reads state.query and state.retrieved_chunks.
    Writes state.answer.
    """
    if state.error:
        logger.warning(
            "[answer_generation_node] Skipping due to upstream error: %s", state.error
        )
        return state

    logger.info(
        "[answer_generation_node] Generating answer (chunks=%d)…",
        len(state.retrieved_chunks),
    )

    try:
        # Derive sector from retrieved chunks when not explicitly set,
        # so the YAML prompt receives a meaningful sector_input value.
        sector = state.sector or (
            state.retrieved_chunks[0].sector if state.retrieved_chunks else ""
        )
        answer = generate_answer(state.query, state.retrieved_chunks, sector=sector)
        state.answer = answer
        log_node_metadata(
            "answer_generation_node",
            query=state.query,
            retrieved_docs_count=len(state.retrieved_chunks),
            extra={
                "answer_len": len(answer),
                "model": settings.llm_model,
            },
        )
    except Exception as exc:
        state.error = f"answer_generation_node failed: {exc}"
        logger.error(state.error)
        state.answer = "An error occurred while generating the answer. Please try again."

    return state
