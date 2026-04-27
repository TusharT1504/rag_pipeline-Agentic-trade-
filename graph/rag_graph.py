"""graph/rag_graph.py

Builds and compiles the LangGraph StateGraph that orchestrates the full
RAG pipeline.

                    ┌──────────────────────┐
                    │  check_embeddings    │
                    └──────────┬───────────┘
                     ┌─────────┴─────────┐
               NO (embed)            YES (retrieve)
                     │                   │
          ┌──────────▼─────────┐         │
          │     load_pdfs      │         │
          └──────────┬─────────┘         │
          ┌──────────▼─────────┐         │
          │  chunk_documents   │         │
          └──────────┬─────────┘         │
          ┌──────────▼─────────┐         │
          │   embed_chunks     │         │
          └──────────┬─────────┘         │
          ┌──────────▼─────────┐         │
          │   store_vectors    │         │
          └──────────┬─────────┘         │
                     └─────────┬─────────┘
                     ┌─────────▼─────────┐
                     │  retrieve_chunks  │
                     └─────────┬─────────┘
                     ┌─────────▼─────────┐
                     │   analyze_tools   │
                     └─────────┬─────────┘
                     ┌─────────▼─────────┐
                     │  generate_answer  │
                     └─────────┬─────────┘
                               END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from graph.state import RAGState
from graph.nodes import (
    check_embeddings_node,
    load_pdfs_node,
    chunk_documents_node,
    embed_chunks_node,
    store_vectors_node,
    retrieve_chunks_node,
    analyze_tools_node,
    generate_answer_node,
)


def _route_after_check(state: RAGState) -> str:
    """Conditional edge: go to embedding pipeline or directly to retrieval."""
    if state.error:
        return END  # type: ignore[return-value]
    return "retrieve_chunks" if state.embeddings_exist else "load_pdfs"


def _check_error(state: RAGState) -> str:
    """Generic error guard used on non-branching edges."""
    return END if state.error else "continue"  # type: ignore[return-value]


def build_rag_graph() -> StateGraph:
    """Construct and compile the RAG LangGraph."""

    graph = StateGraph(RAGState)

    # ── Register nodes ──────────────────────────────────────────────────────
    graph.add_node("check_embeddings", check_embeddings_node)
    graph.add_node("load_pdfs", load_pdfs_node)
    graph.add_node("chunk_documents", chunk_documents_node)
    graph.add_node("embed_chunks", embed_chunks_node)
    graph.add_node("store_vectors", store_vectors_node)
    graph.add_node("retrieve_chunks", retrieve_chunks_node)
    graph.add_node("analyze_tools", analyze_tools_node)
    graph.add_node("generate_answer", generate_answer_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("check_embeddings")

    # ── Conditional branch after embedding check ────────────────────────────
    graph.add_conditional_edges(
        "check_embeddings",
        _route_after_check,
        {
            "load_pdfs": "load_pdfs",
            "retrieve_chunks": "retrieve_chunks",
            END: END,
        },
    )

    # ── Embedding pipeline (linear) ─────────────────────────────────────────
    graph.add_edge("load_pdfs", "chunk_documents")
    graph.add_edge("chunk_documents", "embed_chunks")
    graph.add_edge("embed_chunks", "store_vectors")
    graph.add_edge("store_vectors", "retrieve_chunks")

    # ── Retrieval → analysis → generation → end ─────────────────────────────
    graph.add_edge("retrieve_chunks", "analyze_tools")
    graph.add_edge("analyze_tools", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# Singleton compiled graph
rag_graph = build_rag_graph()
