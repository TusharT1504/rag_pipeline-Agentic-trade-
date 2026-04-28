"""
RAG LangGraph — Conditional graph supporting ingestion and retrieval modes.

Graph structure::

    START
      │
      ▼
    router ──── ingest_flag == True ────► pdf_loader ► chunking ► embedding ► vector_store ► END
      │
      └── ingest_flag == False (default) ► retrieval ► answer_generation ► END

The ``router`` node is a lightweight pass-through that sets state.mode and
returns the routing key consumed by ``add_conditional_edges``.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END, START

from graph.state import RAGState
from observability.langsmith import traced_function
from graph.nodes import (
    pdf_loader_node,
    chunking_node,
    embedding_node,
    vector_store_node,
    retrieval_node,
    answer_generation_node,
)

logger = logging.getLogger(__name__)


# ── Router ────────────────────────────────────────────────────────────────────


@traced_function("router_node", metadata={"component": "graph"})
def router_node(state: RAGState) -> RAGState:
    """
    Routing node: set state.mode based on state.ingest_flag.
    Returns state unchanged (routing happens via conditional edge function).
    """
    state.mode = "ingestion" if state.ingest_flag else "retrieval"
    logger.info("[router_node] Mode selected: %s", state.mode)
    return state


def _route(state: RAGState) -> str:
    """
    Conditional edge function consumed by LangGraph.
    Returns the edge key ("ingestion" or "retrieval").
    """
    return "ingestion" if state.ingest_flag else "retrieval"


# ── Graph builder ─────────────────────────────────────────────────────────────


@traced_function("build_rag_graph", metadata={"component": "graph"})
def build_rag_graph() -> StateGraph:
    """
    Build and compile the conditional RAG LangGraph.

    Returns:
        A compiled ``StateGraph`` ready for ``.invoke()`` or ``.ainvoke()``.
    """
    # Use RAGState as the graph's state schema.
    # LangGraph requires the state class or a TypedDict; we pass RAGState
    # directly and use its dataclass fields as the shared state contract.
    graph = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("router", router_node)

    # Ingestion pipeline
    graph.add_node("pdf_loader", pdf_loader_node)
    graph.add_node("chunking", chunking_node)
    graph.add_node("embedding", embedding_node)
    graph.add_node("vector_store", vector_store_node)

    # Retrieval pipeline: fetch namespace contents directly; no query embedding.
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("answer_generation", answer_generation_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "router")

    # Conditional branch off the router
    graph.add_conditional_edges(
        "router",
        _route,
        {
            "ingestion": "pdf_loader",
            "retrieval": "retrieval",
        },
    )

    # Ingestion pipeline chain
    graph.add_edge("pdf_loader", "chunking")
    graph.add_edge("chunking", "embedding")
    graph.add_edge("embedding", "vector_store")
    graph.add_edge("vector_store", END)

    # Retrieval pipeline chain
    graph.add_edge("retrieval", "answer_generation")
    graph.add_edge("answer_generation", END)

    compiled = graph.compile()
    logger.info("RAG graph compiled successfully.")
    return compiled


# Module-level compiled graph (lazy singleton pattern via module import)
_compiled_graph = None


@traced_function("get_rag_graph", metadata={"component": "graph"})
def get_rag_graph():
    """Return the cached compiled RAG graph (built once per process)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph
