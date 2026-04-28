"""Graph package exports.

Heavy graph construction imports are resolved lazily to avoid circular imports
when tool modules only need state models.
"""

__all__ = [
    "RAGState",
    "DocumentChunk",
    "RetrievedChunk",
    "build_rag_graph",
    "get_rag_graph",
]


def __getattr__(name: str):
    if name in {"RAGState", "DocumentChunk", "RetrievedChunk"}:
        from .state import DocumentChunk, RAGState, RetrievedChunk

        return {
            "RAGState": RAGState,
            "DocumentChunk": DocumentChunk,
            "RetrievedChunk": RetrievedChunk,
        }[name]
    if name in {"build_rag_graph", "get_rag_graph"}:
        from .rag_graph import build_rag_graph, get_rag_graph

        return {
            "build_rag_graph": build_rag_graph,
            "get_rag_graph": get_rag_graph,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
