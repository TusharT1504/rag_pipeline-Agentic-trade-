"""
Answer Generation Tool — Retrieval pipeline, Stage 3.

Loads Sector_analysis_prompt.yaml ONCE at import time and uses it
verbatim as the LLM system prompt.  The retrieved Pinecone chunks and
the user query are injected into the USER message so the prompt file
itself is NEVER modified or re-constructed at runtime.

Flow
----
  system  ← Sector_analysis_prompt.yaml  (loaded as-is, never changed)
  user    ← RETRIEVED_CONTEXT block  +  ANALYST_QUERY block

Groq is called with exponential back-off retry on rate-limit / 5xx.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from groq import Groq, RateLimitError, APIStatusError

from config.settings import get_settings
from graph.state import RetrievedChunk
from observability.langsmith import traced_function, traced_tool

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Resolve prompt file path ──────────────────────────────────────────────────
# Default: <project_root>/prompts/Sector_analysis_prompt.yaml
# Override via SECTOR_PROMPT_PATH in your .env if needed.

_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "Sector_analysis_prompt.yaml"
)


def _resolve_prompt_path() -> Path:
    """Return configured prompt path, falling back when the override is blank."""
    configured_path = getattr(settings, "sector_prompt_path", "")
    if not str(configured_path).strip():
        return _DEFAULT_PROMPT_PATH
    return Path(configured_path).expanduser()


SECTOR_PROMPT_PATH: Path = _resolve_prompt_path()


# ── Load YAML prompt ONCE at module import ────────────────────────────────────

def _load_yaml_prompt(path: Path) -> str:
    """
    Read the YAML prompt file and return its raw text.

    The file is used VERBATIM as the system message — no parsing, no
    reconstruction, no templating.  This guarantees the analyst persona,
    constraints, output format, and quality criteria in the YAML are
    always honoured exactly as written.

    Args:
        path: Absolute path to Sector_analysis_prompt.yaml.

    Returns:
        Raw YAML text string.

    Raises:
        FileNotFoundError: If the file does not exist at *path*.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Sector analysis prompt not found at: {path}\n"
            "Place Sector_analysis_prompt.yaml inside the prompts/ directory "
            "or set SECTOR_PROMPT_PATH in your .env file."
        )
    if path.is_dir():
        raise IsADirectoryError(
            f"SECTOR_PROMPT_PATH points to a directory, not a file: {path}\n"
            "Set it to Sector_analysis_prompt.yaml, or leave it blank to use "
            "prompts/Sector_analysis_prompt.yaml."
        )
    raw = path.read_text(encoding="utf-8")
    logger.info("Sector analysis prompt loaded: %s (%d chars)", path.name, len(raw))
    return raw


# Module-level constant — loaded once, reused for every LLM call.
SYSTEM_PROMPT: str = _load_yaml_prompt(SECTOR_PROMPT_PATH)


# ── Groq client singleton ─────────────────────────────────────────────────────

_groq_client: Groq | None = None


@traced_function("groq_get_client", metadata={"component": "llm"})
def _get_groq_client() -> Groq:
    """Return (and lazily initialise) the module-level Groq client."""
    global _groq_client
    if _groq_client is None:
        logger.info("Initialising Groq client (model=%s)…", settings.llm_model)
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


# ── User message builder ──────────────────────────────────────────────────────

@traced_function("build_answer_user_message", metadata={"component": "prompt"})
def _build_user_message(
    query: str,
    chunks: list[RetrievedChunk],
    sector: str,
) -> str:
    """
    Build the user-turn message paired with the YAML system prompt.

    The YAML defines: role, rules, output structure, quality criteria.
    This function fulfils the DATA_INPUT_INFORMATION and CONTEXT fields
    that the YAML references by injecting two concrete blocks:

    Block 1 — RETRIEVED_CONTEXT
        The Pinecone chunks that act as the "Uploaded PDF documents"
        the prompt refers to.  Each chunk includes its source file,
        page number, sector, and similarity score so the model can
        cite evidence precisely.

    Block 2 — ANALYST_QUERY
        The specific question or analysis task for this call.
        Maps to the sector_input / task fields in the YAML CONTEXT block.

    Final message shape seen by the LLM
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    system = Sector_analysis_prompt.yaml  (role + rules + output format)
    user   = retrieved chunks  +  analyst query

    Args:
        query:  Analyst's question or report request.
        chunks: Pinecone-retrieved context chunks (the grounding evidence).
        sector: Sector / namespace label (e.g. "ESDM", "cement").

    Returns:
        Formatted string for the user role message.
    """
    # Numbered, clearly labelled context entries
    context_entries: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        context_entries.append(
            f"[CHUNK {i}]\n"
            f"  SOURCE FILE : {chunk.source_file}\n"
            f"  PAGE        : {chunk.page_number}\n"
            f"  SECTOR      : {chunk.sector}\n"
            f"  SIMILARITY  : {chunk.score:.4f}\n"
            f"  CONTENT     :\n{chunk.text}"
        )

    context_block = "\n\n" + ("-" * 60) + "\n\n".join(context_entries)

    user_message = (
        "## DATA_INPUT_INFORMATION\n\n"
        f"### sector_input\n{sector}\n\n"
        "### source_material\n"
        "The passages below have been retrieved from the sector knowledge base "
        f"(Pinecone namespace: '{sector}').  "
        "Treat them as the PDF content referenced in the prompt under "
        "DATA_INPUT_INFORMATION.  "
        "Use ONLY this content — do NOT introduce any external knowledge.\n\n"
        f"### RETRIEVED_CONTEXT{context_block}\n\n"
        + ("-" * 60) + "\n\n"
        "## ANALYST_QUERY\n"
        f"{query}\n\n"
        + ("-" * 60) + "\n\n"
        "Produce the full structured report exactly as specified in OUTPUT_FORMAT.\n"
        "Every major claim MUST cite the SOURCE FILE and PAGE from RETRIEVED_CONTEXT."
    )

    return user_message


# ── Main generation function ──────────────────────────────────────────────────

@traced_tool("generate_answer", metadata={"pipeline": "retrieval", "stage": "answer_generation"})
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    sector: str = "",
    max_retries: int | None = None,
) -> str:
    """
    Generate a sector analysis report using the Groq LLM.

    System prompt = Sector_analysis_prompt.yaml  (loaded once at import,
                    used verbatim — never reconstructed).
    User message  = Pinecone-retrieved context chunks  +  analyst query.

    Retry strategy: exponential back-off on RateLimitError and Groq 5xx.
    Fallback:       plain-language message when no chunks are available.

    Args:
        query:       The analyst's question or report request.
        chunks:      Pinecone context chunks (the grounding evidence).
        sector:      Sector / namespace label (e.g. "ESDM", "cement").
                     Injected into the user message as sector_input.
        max_retries: Override settings.llm_max_retries if provided.

    Returns:
        The LLM-generated sector analysis report as a string.
    """
    retries = max_retries if max_retries is not None else settings.llm_max_retries
    client = _get_groq_client()

    # ── Fallback: no chunks retrieved ────────────────────────────────────────
    if not chunks:
        fallback = (
            "No relevant context was found in the knowledge base for your query.\n\n"
            "Possible reasons:\n"
            f"  • Pinecone namespace '{sector}' may be empty — run the ingestion "
            "pipeline first (POST /ingest or `python main.py ingest`).\n"
            "  • The query may not match any indexed content — try rephrasing.\n"
            "  • The similarity threshold may be too high — lower "
            "RETRIEVAL_SCORE_THRESHOLD in your .env."
        )
        logger.warning(
            "generate_answer: 0 chunks for query='%s' sector='%s' — returning fallback.",
            query[:80],
            sector,
        )
        return fallback

    # ── Build messages ────────────────────────────────────────────────────────
    user_message = _build_user_message(query, chunks, sector or "unknown")

    logger.info(
        "Calling Groq | model=%s | sector=%s | chunks=%d | "
        "system_chars=%d | user_chars=%d",
        settings.llm_model,
        sector,
        len(chunks),
        len(SYSTEM_PROMPT),
        len(user_message),
    )

    # ── Groq call with exponential back-off retry ─────────────────────────────
    last_exc: Exception | None = None

    for attempt in range(1, retries + 2):  # attempt 1 … retries+1
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            answer: str = response.choices[0].message.content or ""
            logger.info(
                "Report generated | attempt=%d | tokens=%s | answer_chars=%d",
                attempt,
                getattr(response.usage, "total_tokens", "n/a"),
                len(answer),
            )
            return answer.strip()

        except RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning(
                "Groq rate-limit (attempt %d/%d) — retrying in %ds…",
                attempt, retries + 1, wait,
            )
            last_exc = exc
            time.sleep(wait)

        except APIStatusError as exc:
            if exc.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "Groq 5xx %d (attempt %d/%d) — retrying in %ds…",
                    exc.status_code, attempt, retries + 1, wait,
                )
                last_exc = exc
                time.sleep(wait)
            else:
                logger.error("Groq client error %d: %s", exc.status_code, exc)
                raise

        except Exception as exc:
            logger.error("Unexpected Groq error: %s", exc)
            raise

    raise RuntimeError(
        f"Groq API failed after {retries + 1} attempt(s). Last error: {last_exc}"
    )
