"""tools/answer_generation_tool.py

Takes retrieved chunks and the original query; returns a structured JSON
answer with sources and a confidence estimate.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from config import settings
from observability import compact_search_outputs, traceable

logger = logging.getLogger(__name__)

# ── Provider-specific client helpers ────────────────────────────────────────

_CONFIDENCE_VALUES = {"high", "medium", "low"}

_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "sources", "confidence"],
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["document_name", "page_number", "section"],
                "properties": {
                    "document_name": {"type": "string"},
                    "page_number": {"type": "integer"},
                    "section": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string", "enum": sorted(_CONFIDENCE_VALUES)},
    },
}


def _google_response_schema(gtypes: Any) -> Any:
    """Build a Google SDK response schema object for strict structured output."""
    return gtypes.Schema(
        type=gtypes.Type.OBJECT,
        required=["answer", "sources", "confidence"],
        properties={
            "answer": gtypes.Schema(type=gtypes.Type.STRING),
            "sources": gtypes.Schema(
                type=gtypes.Type.ARRAY,
                items=gtypes.Schema(
                    type=gtypes.Type.OBJECT,
                    required=["document_name", "page_number", "section"],
                    properties={
                        "document_name": gtypes.Schema(type=gtypes.Type.STRING),
                        "page_number": gtypes.Schema(type=gtypes.Type.INTEGER),
                        "section": gtypes.Schema(type=gtypes.Type.STRING),
                    },
                ),
            ),
            "confidence": gtypes.Schema(
                type=gtypes.Type.STRING,
                enum=sorted(_CONFIDENCE_VALUES),
            ),
        },
    )


def _parse_json_string(raw: str) -> dict[str, Any]:
    """Parse a JSON string into a dict; fallback to a low-confidence answer."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON: %s", raw[:200])
        return {"answer": raw, "sources": [], "confidence": "low"}

    if isinstance(parsed, dict):
        return parsed

    logger.error("LLM returned non-object JSON payload: %s", type(parsed).__name__)
    return {
        "answer": "The information is not available in the provided documents.",
        "sources": [],
        "confidence": "low",
    }


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Guarantee required output keys and expected value types."""
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = "The information is not available in the provided documents."
    else:
        answer = answer.strip()

    sources: list[dict[str, Any]] = []
    raw_sources = result.get("sources", [])
    if isinstance(raw_sources, list):
        for src in raw_sources:
            if not isinstance(src, dict):
                continue
            document_name = src.get("document_name")
            section = src.get("section")
            page_number = src.get("page_number")

            if not isinstance(document_name, str) or not document_name.strip():
                document_name = "unknown"
            if not isinstance(section, str) or not section.strip():
                section = "General"
            try:
                page_number = int(page_number)
            except (TypeError, ValueError):
                page_number = 0

            sources.append(
                {
                    "document_name": document_name,
                    "page_number": page_number,
                    "section": section,
                }
            )

    confidence = result.get("confidence", "low")
    confidence = confidence.lower() if isinstance(confidence, str) else "low"
    if confidence not in _CONFIDENCE_VALUES:
        confidence = "low"

    return {"answer": answer, "sources": sources, "confidence": confidence}


def _compact_answer_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    chunks = inputs.get("retrieved_chunks")
    return {
        "query": inputs.get("query", ""),
        "retrieval": compact_search_outputs(chunks),
    }


def _compact_answer_outputs(output: Any) -> dict[str, Any]:
    result = output if isinstance(output, dict) else {}
    answer = result.get("answer", "")
    return {
        "answer_preview": answer[:300] if isinstance(answer, str) else "",
        "source_count": len(result.get("sources", [])) if isinstance(result.get("sources"), list) else 0,
        "confidence": result.get("confidence"),
    }


def _compact_llm_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    system_prompt = inputs.get("system_prompt", "")
    user_message = inputs.get("user_message", "")
    return {
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL,
        "system_prompt_chars": len(system_prompt) if isinstance(system_prompt, str) else 0,
        "user_message_chars": len(user_message) if isinstance(user_message, str) else 0,
    }


@traceable(
    run_type="llm",
    name="Generate Answer LLM",
    metadata={
        "provider": settings.LLM_PROVIDER,
        "ls_provider": settings.LLM_PROVIDER,
        "ls_model_name": settings.LLM_MODEL,
    },
    process_inputs=_compact_llm_inputs,
    process_outputs=_compact_answer_outputs,
)
def _call_llm(system_prompt: str, user_message: str) -> dict[str, Any]:
    """Route to the configured provider and return a structured dict response."""
    provider = settings.LLM_PROVIDER

    if provider == "google":
        from google import genai
        from google.genai import types as gtypes
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=user_message,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_google_response_schema(gtypes),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "model_dump"):
                parsed = parsed.model_dump()
            elif hasattr(parsed, "dict"):
                parsed = parsed.dict()

            if isinstance(parsed, dict):
                return parsed

        return _parse_json_string(response.text or "{}")

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use 'google'")


_SYSTEM_PROMPT = """\
You are a precise document-analysis assistant.
You will be given:
  1. A user question.
  2. A numbered list of retrieved document excerpts, each with metadata.

Your task:
  • Answer the question using ONLY the provided excerpts.
  • If the information is absent, respond with the exact phrase:
    "The information is not available in the provided documents."
  • Never hallucinate facts or mix in outside knowledge.

Return a single valid JSON object — no markdown fences, no extra text:
{
  "answer": "<clear and precise answer>",
  "sources": [
    {
      "document_name": "<filename>",
      "page_number": <int>,
      "section": "<section name>"
    }
  ],
  "confidence": "high | medium | low"
}

Confidence guidance:
  • high   — multiple relevant excerpts directly address the question.
  • medium — partial or indirect evidence.
  • low    — very weak evidence; answer may be incomplete.
"""


@traceable(
    run_type="chain",
    name="Generate Structured Answer",
    metadata={"provider": settings.LLM_PROVIDER, "model": settings.LLM_MODEL},
    process_inputs=_compact_answer_inputs,
    process_outputs=_compact_answer_outputs,
)
def answer_generation_tool(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a structured answer from retrieved context.

    Parameters
    ----------
    query : str
        Original user question.
    retrieved_chunks : list[dict]
        Output of retrieval_tool — each has ``text``, ``score``, ``metadata``.

    Returns
    -------
    dict
        {"answer": str, "sources": list[dict], "confidence": str}
    """
    if not retrieved_chunks:
        return {
            "answer": "The information is not available in the provided documents.",
            "sources": [],
            "confidence": "low",
        }

    # Build context block
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk["metadata"]
        context_lines.append(
            f"[{i}] Document: {meta.get('document_name', 'unknown')} | "
            f"Page: {meta.get('page_number', '?')} | "
            f"Section: {meta.get('section', 'General')} | "
            f"Score: {chunk['score']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_lines)
    user_message = f"Question: {query}\n\nContext:\n{context}"

    logger.info("Calling LLM (%s / %s) for answer generation …", settings.LLM_PROVIDER, settings.LLM_MODEL)
    result = _call_llm(_SYSTEM_PROMPT, user_message)
    return _normalize_result(result)
