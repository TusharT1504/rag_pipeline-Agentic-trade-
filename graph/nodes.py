"""graph/nodes.py

LangGraph node functions.  Each function:
  • Receives the current RAGState
  • Calls one or more tools
  • Returns a *partial* dict that LangGraph merges back into the state

Embedding and retrieval tools are chosen at import-time based on
settings.EMBEDDING_PROVIDER so the rest of the graph is unchanged.
"""

from __future__ import annotations
import logging
from typing import Any

from config import settings
from graph.state import RAGState
from tools import (
    pdf_loader_tool,
    chunking_tool,
    index_exists,
    upsert_chunks,
    answer_generation_tool,
    tool_analysis_tool,
)

# ── Dynamic provider selection ───────────────────────────────────────────────
_provider = settings.EMBEDDING_PROVIDER.lower()

if _provider == "google":
    from tools.embedding_tool_google import embedding_tool
    from tools.retrieval_tool_google import retrieval_tool
elif _provider == "sentence_transformers":
    from tools.embedding_tool_st import embedding_tool
    from tools.retrieval_tool_st import retrieval_tool

logger = logging.getLogger(__name__)


# ── Node 1 ─────────────────────────────────────────────────────────────────
def check_embeddings_node(state: RAGState) -> dict[str, Any]:
    """Decide whether embeddings need to be (re-)computed."""
    logger.info("NODE: check_embeddings")
    try:
        exists = index_exists()
        return {"embeddings_exist": exists}
    except Exception as exc:
        logger.error("check_embeddings failed: %s", exc)
        return {"error": str(exc)}


# ── Node 2 ─────────────────────────────────────────────────────────────────
def load_pdfs_node(state: RAGState) -> dict[str, Any]:
    """Load raw text from every PDF in the configured directory."""
    logger.info("NODE: load_pdfs")
    try:
        pages = pdf_loader_tool()
        return {"raw_texts": pages}
    except Exception as exc:
        logger.error("load_pdfs failed: %s", exc)
        return {"error": str(exc)}


# ── Node 3 ─────────────────────────────────────────────────────────────────
def chunk_documents_node(state: RAGState) -> dict[str, Any]:
    """Split raw pages into overlapping semantic chunks."""
    logger.info("NODE: chunk_documents (%d pages)", len(state.raw_texts))
    try:
        chunks = chunking_tool(state.raw_texts)
        return {"chunks": chunks}
    except Exception as exc:
        logger.error("chunk_documents failed: %s", exc)
        return {"error": str(exc)}


# ── Node 4 ─────────────────────────────────────────────────────────────────
def embed_chunks_node(state: RAGState) -> dict[str, Any]:
    """Compute embeddings for all chunks."""
    logger.info("NODE: embed_chunks (%d chunks)", len(state.chunks))
    try:
        embedded = embedding_tool(state.chunks)
        return {"chunks": embedded}
    except Exception as exc:
        logger.error("embed_chunks failed: %s", exc)
        return {"error": str(exc)}


# ── Node 5 ─────────────────────────────────────────────────────────────────
def store_vectors_node(state: RAGState) -> dict[str, Any]:
    """Upsert embedded chunks into Pinecone."""
    logger.info("NODE: store_vectors (%d chunks)", len(state.chunks))
    try:
        count = upsert_chunks(state.chunks)
        logger.info("Stored %d vectors in Pinecone.", count)
        return {"vectors_stored": True, "embeddings_exist": True}
    except Exception as exc:
        logger.error("store_vectors failed: %s", exc)
        return {"error": str(exc)}


# ── Node 6 ─────────────────────────────────────────────────────────────────
def retrieve_chunks_node(state: RAGState) -> dict[str, Any]:
    """Retrieve the top-k most relevant chunks for the user query."""
    logger.info("NODE: retrieve_chunks — query='%s'", state.query[:80])
    try:
        chunks = retrieval_tool(state.query)
        return {"retrieved_chunks": chunks}
    except Exception as exc:
        logger.error("retrieve_chunks failed: %s", exc)
        return {"error": str(exc)}


# ── Node 7 ─────────────────────────────────────────────────────────────────
def analyze_tools_node(state: RAGState) -> dict[str, Any]:
    """Summarize retrieval/tool behavior for debugging and LangSmith traces."""
    logger.info("NODE: analyze_tools (%d chunks)", len(state.retrieved_chunks))
    try:
        analysis = tool_analysis_tool(state.query, state.retrieved_chunks)
        return {"tool_analysis": analysis}
    except Exception as exc:
        logger.error("analyze_tools failed: %s", exc)
        return {"error": str(exc)}


# ── Node 8 ─────────────────────────────────────────────────────────────────
def generate_answer_node(state: RAGState) -> dict[str, Any]:
    """Generate the final structured answer from retrieved context."""
    logger.info("NODE: generate_answer (%d chunks)", len(state.retrieved_chunks))
    try:
        answer = answer_generation_tool(state.query, state.retrieved_chunks)
        return {"final_answer": answer}
    except Exception as exc:
        logger.error("generate_answer failed: %s", exc)
        return {"error": str(exc)}
