from __future__ import annotations

from typing import Iterable, List

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from .settings import SETTINGS, Settings


class EmbeddingsConfigError(RuntimeError):
    """Raised when the embedding provider configuration is invalid."""


class BGEM3Embeddings:
    """Shim around sentence-transformers for BGE-M3."""

    def __init__(self, model_name: str):
        try:
            self._model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - network/env issues
            raise EmbeddingsConfigError(
                f"Failed to load SentenceTransformer model '{model_name}': {exc}"
            ) from exc

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embeddings(settings: Settings | None = None):
    """Return an embedding model instance that matches the settings."""

    cfg = settings or SETTINGS

    if cfg.embeddings_provider == "azure_openai":
        missing = [
            name
            for name, value in [
                ("AZURE_OPENAI_ENDPOINT", cfg.azure_openai_endpoint),
                ("AZURE_OPENAI_API_VERSION", cfg.azure_openai_api_version),
                ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", cfg.azure_openai_embedding_deployment),
                ("AZURE_OPENAI_EMBEDDING_API_KEY", cfg.azure_openai_embedding_api_key),
            ]
            if not value
        ]
        if missing:
            raise EmbeddingsConfigError(
                "Azure OpenAI embeddings require environment variables: "
                + ", ".join(missing)
            )

        return AzureOpenAIEmbeddings(
            model=cfg.embeddings_model,
            deployment=cfg.azure_openai_embedding_deployment,
            azure_endpoint=cfg.azure_openai_endpoint,
            openai_api_version=cfg.azure_openai_api_version,
            openai_api_key=cfg.azure_openai_embedding_api_key,
        )

    if cfg.embeddings_provider == "openai":
        missing = [
            name
            for name, value in [
                ("OPENAI_BASE_URL", cfg.openai_api_base),
                ("OPENAI_EMBEDDING_MODEL", cfg.openai_embedding_model or cfg.embeddings_model),
                ("OPENAI_EMBEDDING_API_KEY", cfg.openai_embedding_api_key),
            ]
            if not value
        ]
        if missing:
            raise EmbeddingsConfigError(
                "OpenAI-compatible embeddings require environment variables: "
                + ", ".join(missing)
            )

        return OpenAIEmbeddings(
            base_url=cfg.openai_api_base,
            api_key=cfg.openai_embedding_api_key,
            model=cfg.openai_embedding_model or cfg.embeddings_model,
            model_kwargs={"encoding_format": "float"},
        )

    if cfg.embeddings_provider == "bge":
        return BGEM3Embeddings(cfg.embeddings_model)

    raise EmbeddingsConfigError(
        f"Unsupported embeddings provider: {cfg.embeddings_provider}"
    )
