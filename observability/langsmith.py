"""
LangSmith observability and tracing utilities.
Wraps every node call with structured logging of inputs, outputs,
execution time, and domain metadata.
"""

import time
import logging
import functools
from dataclasses import fields, is_dataclass
from typing import Any, Callable, TypeVar
from datetime import datetime, timezone

from langsmith import Client
from langsmith.run_helpers import traceable

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── LangSmith client (lazy) ──────────────────────────────────────────────────

_ls_client: Client | None = None


def get_langsmith_client() -> Client | None:
    """Return a cached LangSmith client, or None if not configured."""
    global _ls_client
    if _ls_client is not None:
        return _ls_client
    if not settings.langchain_api_key or not settings.langchain_tracing_v2:
        logger.warning("LANGCHAIN_API_KEY not set — LangSmith tracing disabled.")
        return None
    try:
        _ls_client = Client(
            api_url=settings.langchain_endpoint,
            api_key=settings.langchain_api_key,
        )
        logger.info("LangSmith client initialised (project=%s)", settings.langchain_project)
    except Exception as exc:
        logger.error("Failed to create LangSmith client: %s", exc)
        _ls_client = None
    return _ls_client


# ── Trace decorator ──────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])


def _tracing_enabled() -> bool:
    return bool(settings.langchain_api_key and settings.langchain_tracing_v2)


def _compact_value(value: Any) -> Any:
    """Return a LangSmith-friendly summary without large vectors or full documents."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= 300:
            return value
        return {"type": "str", "chars": len(value), "preview": value[:300]}
    if isinstance(value, dict):
        return {
            str(k): _compact_value(v)
            for k, v in list(value.items())[:20]
            if str(k).lower() not in {"values", "embedding", "vector", "query_vector"}
        }
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, float)) for item in value):
            return {"type": "vector", "dimension": len(value)}
        if value and all(
            isinstance(item, (list, tuple))
            and all(isinstance(v, (int, float)) for v in item)
            for item in value[:5]
        ):
            first = value[0] if value else []
            return {"type": "vector_batch", "count": len(value), "dimension": len(first)}
        return {
            "type": type(value).__name__,
            "count": len(value),
            "sample": [_compact_value(item) for item in list(value)[:3]],
        }
    if is_dataclass(value):
        summary: dict[str, Any] = {"type": type(value).__name__}
        for field in fields(value):
            if field.name in {"text", "query_embedding", "chunk_embeddings"}:
                field_value = getattr(value, field.name)
                summary[field.name] = _compact_value(field_value)
            elif field.name in {
                "chunk_id",
                "sector",
                "source_file",
                "page_number",
                "namespace",
                "score",
                "query",
                "namespaces",
                "top_k",
                "error",
                "mode",
                "upserted_count",
            }:
                summary[field.name] = _compact_value(getattr(value, field.name))
        return summary
    return {"type": type(value).__name__, "repr": repr(value)[:300]}


def _compact_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: _compact_value(value) for key, value in inputs.items()}


def _compact_trace_outputs(output: Any) -> Any:
    return _compact_value(output)


def traced_function(
    run_name: str,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Trace a regular helper/function with compact LangSmith payloads."""
    extra_meta = metadata or {}

    def decorator(fn: F) -> F:
        if _tracing_enabled():
            return traceable(
                name=run_name,
                run_type=run_type,
                metadata={
                    "project": settings.langchain_project,
                    **extra_meta,
                },
                process_inputs=_compact_trace_inputs,
                process_outputs=_compact_trace_outputs,
            )(fn)  # type: ignore[return-value]
        return fn

    return decorator


def traced_tool(run_name: str, metadata: dict[str, Any] | None = None) -> Callable[[F], F]:
    """Trace a pipeline tool function as a LangSmith tool run."""
    tool_metadata = {"component": "tool", **(metadata or {})}
    return traced_function(run_name, run_type="tool", metadata=tool_metadata)


def traced_node(run_name: str, metadata: dict[str, Any] | None = None) -> Callable[[F], F]:
    """
    Decorator that wraps an async or sync node function with:
    - LangSmith tracing via @traceable
    - Structured logging of inputs, outputs, and wall-clock time
    - Custom metadata attached to the run

    Usage::

        @traced_node("retrieval_node", metadata={"stage": "retrieval"})
        async def retrieval_node(state: RAGState) -> RAGState:
            ...
    """
    extra_meta = metadata or {}

    def decorator(fn: F) -> F:
        # Apply LangSmith @traceable only when tracing is actually configured.
        if _tracing_enabled():
            traced_fn = traceable(
                name=run_name,
                run_type="chain",
                metadata={
                    "project": settings.langchain_project,
                    "node": run_name,
                    **extra_meta,
                },
                process_inputs=_compact_trace_inputs,
                process_outputs=_compact_trace_outputs,
            )(fn)
        else:
            traced_fn = fn

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(
                "[%s] START | args_keys=%s kwargs_keys=%s",
                run_name,
                [type(a).__name__ for a in args],
                list(kwargs.keys()),
            )
            try:
                result = await traced_fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[%s] END | elapsed=%.3fs", run_name, elapsed)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error("[%s] ERROR after %.3fs | %s", run_name, elapsed, exc)
                raise

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info("[%s] START", run_name)
            try:
                result = traced_fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[%s] END | elapsed=%.3fs", run_name, elapsed)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error("[%s] ERROR after %.3fs | %s", run_name, elapsed, exc)
                raise

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def log_node_metadata(
    run_name: str,
    query: str | None = None,
    namespace: str | list[str] | None = None,
    retrieved_docs_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Emit a structured log line with node-level metadata.
    Call this INSIDE a node after results are available.
    """
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "node": run_name,
    }
    if query is not None:
        payload["query"] = query[:200]  # truncate for log brevity
    if namespace is not None:
        payload["namespace"] = namespace
    if retrieved_docs_count is not None:
        payload["retrieved_docs_count"] = retrieved_docs_count
    if extra:
        payload.update(extra)

    logger.info("[%s] METADATA | %s", run_name, payload)
