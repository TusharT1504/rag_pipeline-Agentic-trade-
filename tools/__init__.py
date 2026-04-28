"""Tools package exports.

Tool implementations are imported lazily so one tool can be imported without
initialising the whole pipeline and tripping circular imports.
"""

__all__ = [
    "load_pdfs",
    "chunk_pages",
    "embed_query",
    "embed_document_chunks",
    "upsert_chunks",
    "query_namespaces",
    "fetch_namespaces_chunks",
    "retrieve_chunks",
    "fetch_namespace_context",
    "generate_answer",
    "analyse_chunks",
    "analyse_retrieved_chunks",
]


def __getattr__(name: str):
    if name == "load_pdfs":
        from .pdf_loader_tool import load_pdfs

        return load_pdfs
    if name == "chunk_pages":
        from .chunking_tool import chunk_pages

        return chunk_pages
    if name in {"embed_query", "embed_document_chunks"}:
        from .embedding_tool import embed_document_chunks, embed_query

        return {
            "embed_query": embed_query,
            "embed_document_chunks": embed_document_chunks,
        }[name]
    if name in {"upsert_chunks", "query_namespaces", "fetch_namespaces_chunks"}:
        from .vector_store_tool import fetch_namespaces_chunks, query_namespaces, upsert_chunks

        return {
            "upsert_chunks": upsert_chunks,
            "query_namespaces": query_namespaces,
            "fetch_namespaces_chunks": fetch_namespaces_chunks,
        }[name]
    if name in {"retrieve_chunks", "fetch_namespace_context"}:
        from .retrieval_tool import fetch_namespace_context, retrieve_chunks

        return {
            "retrieve_chunks": retrieve_chunks,
            "fetch_namespace_context": fetch_namespace_context,
        }[name]
    if name == "generate_answer":
        from .answer_generation_tool import generate_answer

        return generate_answer
    if name in {"analyse_chunks", "analyse_retrieved_chunks"}:
        from .tool_analysis import analyse_chunks, analyse_retrieved_chunks

        return {
            "analyse_chunks": analyse_chunks,
            "analyse_retrieved_chunks": analyse_retrieved_chunks,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
