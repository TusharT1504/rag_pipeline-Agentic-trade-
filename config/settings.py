"""config/settings.py — Centralised settings loaded from .env"""

import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # ── API Keys ────────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str   = os.getenv("GOOGLE_API_KEY", "")
    PINECONE_API_KEY: str  = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str  = os.getenv("PINECONE_INDEX_NAME", "rag-documents")

    # ── Paths ───────────────────────────────────────────────────────────────
    PDF_DIR: str = os.getenv("PDF_DIR", "./pdfs")

    # ── Chunking ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

    # ── Retrieval ───────────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", 13))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    QUIET_THIRD_PARTY_LOGS: bool = _env_bool("QUIET_THIRD_PARTY_LOGS", True)

    # ── Embedding provider ──────────────────────────────────────────────────
    # Options: "google" | "sentence_transformers"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "google")

    # Model name is interpreted by the chosen provider:
    #   google                → "models/text-embedding-004"
    #   sentence_transformers → "all-MiniLM-L6-v2"  (or any HF model name)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

    # Pinecone index dimension — MUST match your embedding model:
    #   Google text-embedding-004       → 768
    #   all-MiniLM-L6-v2               → 384
    #   all-mpnet-base-v2              → 768
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 768))

    # ── LLM ─────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")   # "google"
    LLM_MODEL: str    = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    # ── LangSmith tracing ──────────────────────────────────────────────────
    LANGSMITH_TRACING: bool = (
        _env_bool("LANGSMITH_TRACING", False)
        or _env_bool("LANGSMITH_TRACING_V2", False)
    )
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "rag-pipeline")
    LANGSMITH_ENDPOINT: str = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )

    def __init__(self):
        self.EMBEDDING_PROVIDER = self.EMBEDDING_PROVIDER.strip().lower()
        self.EMBEDDING_MODEL = self.EMBEDDING_MODEL.strip()
        self.LLM_PROVIDER = self.LLM_PROVIDER.strip().lower()
        self.LLM_MODEL = self.LLM_MODEL.strip()
        self.LOG_LEVEL = self.LOG_LEVEL.strip().upper()
        self.LANGSMITH_PROJECT = self.LANGSMITH_PROJECT.strip()
        self.LANGSMITH_ENDPOINT = self.LANGSMITH_ENDPOINT.strip()

        inferred_dimension = self.infer_embedding_dimension()
        if inferred_dimension is not None:
            self.EMBEDDING_DIMENSION = inferred_dimension

        self.configure_langsmith_environment()

    def configure_langsmith_environment(self) -> None:
        """Keep LangSmith's current and legacy tracing env vars in sync."""
        if not self.LANGSMITH_TRACING:
            return

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", self.LANGSMITH_PROJECT)
        os.environ.setdefault("LANGSMITH_ENDPOINT", self.LANGSMITH_ENDPOINT)

    def infer_embedding_dimension(self) -> int | None:
        """Return the expected vector size for known embedding models."""
        model = self.EMBEDDING_MODEL.lower().removeprefix("models/")

        if self.EMBEDDING_PROVIDER == "google":
            return 768

        if self.EMBEDDING_PROVIDER == "sentence_transformers":
            sentence_transformer_dimensions = {
                "all-minilm-l6-v2": 384,
                "all-mpnet-base-v2": 768,
                "multi-qa-minilm-l6-cos-v1": 384,
                "baai/bge-small-en-v1.5": 384,
            }
            return sentence_transformer_dimensions.get(model)

    # ── Validation ──────────────────────────────────────────────────────────
    def validate(self):
        missing = []

        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")

        if self.EMBEDDING_PROVIDER == "google" or self.LLM_PROVIDER == "google":
            if not self.GOOGLE_API_KEY:
                missing.append("GOOGLE_API_KEY")

        if self.LANGSMITH_TRACING and not self.LANGSMITH_API_KEY:
            missing.append("LANGSMITH_API_KEY")

        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example → .env and fill in the values."
            )
        return self


settings = Settings()
