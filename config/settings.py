"""
Configuration settings for the RAG system.
All values are configurable via environment variables.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import AliasChoices, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── LLM ─────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    llm_model: str = Field(default="llama-3.3-70b-versatile", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    llm_max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")

    # ── Pinecone ─────────────────────────────────────────────────────────────
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(..., env="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=11, env="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float = Field(
        default=0.3, env="RETRIEVAL_SCORE_THRESHOLD"
    )
    namespace_fetch_limit: int = Field(
        default=0,
        env="NAMESPACE_FETCH_LIMIT",
    )
    """
    Max records to fetch per namespace when bypassing query embedding.
    0 means fetch every vector ID listed in the namespace.
    """
    default_namespaces: list[str] = Field(
        default=["ESDM", "cement"],
        env="DEFAULT_NAMESPACES",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"),
    )
    langchain_tracing_v2: bool = Field(
        default=True,
        validation_alias=AliasChoices("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2"),
    )
    langchain_project: str = Field(
        default="rag-system",
        validation_alias=AliasChoices("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT"),
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices("LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"),
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=False, env="API_RELOAD")

    # ── Prompt ────────────────────────────────────────────────────────────────
    sector_prompt_path: str = Field(
        default="",
        env="SECTOR_PROMPT_PATH",
    )
    """
    Optional override path to Sector_analysis_prompt.yaml.
    Defaults to <project_root>/prompts/Sector_analysis_prompt.yaml when empty.
    """

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        # Allow list parsing from comma-separated env vars
        json_schema_extra = {"env_nested_delimiter": "__"}

    def model_post_init(self, __context):
        """Set LangSmith env vars so LangChain picks them up automatically."""
        tracing_enabled = self.langchain_tracing_v2 and bool(self.langchain_api_key)
        if self.langchain_api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", self.langchain_api_key)
            os.environ.setdefault("LANGSMITH_API_KEY", self.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", str(tracing_enabled).lower())
        os.environ.setdefault("LANGSMITH_TRACING", str(tracing_enabled).lower())
        os.environ.setdefault("LANGCHAIN_PROJECT", self.langchain_project)
        os.environ.setdefault("LANGSMITH_PROJECT", self.langchain_project)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", self.langchain_endpoint)
        os.environ.setdefault("LANGSMITH_ENDPOINT", self.langchain_endpoint)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
