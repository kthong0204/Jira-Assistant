from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class Settings:
    """Container for tuning the demo without touching code."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "source_docs"
    )
    vector_store_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "vectorstore"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "4"))
    retriever_score_threshold: float | None = (
        float(threshold) if (threshold := os.getenv("RETRIEVER_SCORE_THRESHOLD")) else None
    )
    llm_provider: Literal["azure_openai", "openai"] = os.getenv(
        "LLM_PROVIDER", "azure_openai"
    )  # type: ignore[assignment]
    azure_openai_llm_endpoint: str | None = os.getenv(
        "AZURE_OPENAI_LLM_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    azure_openai_llm_api_version: str | None = os.getenv(
        "AZURE_OPENAI_LLM_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION")
    )
    azure_openai_llm_deployment: str | None = os.getenv(
        "AZURE_OPENAI_LLM_DEPLOYMENT",
        os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")),
    )
    azure_openai_llm_api_key: str | None = os.getenv(
        "AZURE_OPENAI_LLM_API_KEY",
        os.getenv("AZURE_OPENAI_API_KEY"),
    )
    embeddings_provider: Literal["azure_openai", "bge"] = os.getenv(
        "EMBEDDINGS_PROVIDER", "azure_openai"
    )  # type: ignore[assignment]
    embeddings_model: str = os.getenv(
        "EMBEDDINGS_MODEL", "text-embedding-3-small"
    )
    azure_openai_endpoint: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str | None = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_openai_embedding_deployment: str | None = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    azure_openai_embedding_api_key: str | None = os.getenv(
        "AZURE_OPENAI_EMBEDDING_API_KEY",
        os.getenv("AZURE_OPENAI_API_KEY"),
    )
    openai_api_base: str | None = os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str | None = os.getenv("OPENAI_MODEL")
    openai_embedding_model: str | None = os.getenv(
        "OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_MODEL")
    )
    openai_embedding_api_key: str | None = os.getenv(
        "OPENAI_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY")
    )
    jira_base_url: str | None = os.getenv("JIRA_URL")
    jira_email: str | None = os.getenv("JIRA_EMAIL")
    jira_api_token: str | None = os.getenv("JIRA_API_TOKEN")
    jira_bearer_token: str | None = os.getenv("JIRA_BEARER_TOKEN")
    jira_api_path: str = os.getenv("JIRA_API_PATH", "/rest/api/2")
    jira_dataset_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "jira_samples.json"
    )
    jira_search_endpoint: str = os.getenv("JIRA_SEARCH_ENDPOINT", "/search")
    jira_search_method: str = os.getenv("JIRA_SEARCH_METHOD", "GET")
    git_repo_root: Path = field(
        default_factory=lambda: Path(os.getenv("GIT_REPO_ROOT", str(PROJECT_ROOT))).resolve()
    )

    @property
    def vector_store_path(self) -> Path:
        return self.vector_store_dir / "chroma_index"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()


def with_overrides(**overrides) -> Settings:
    clean = {k: v for k, v in overrides.items() if v is not None}
    if not clean:
        return SETTINGS
    return replace(SETTINGS, **clean)
