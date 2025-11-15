from __future__ import annotations

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .settings import SETTINGS, Settings


class LLMConfigError(RuntimeError):
    """Raised when LLM configuration/env is invalid."""


def get_llm(settings: Settings | None = None):
    cfg = settings or SETTINGS

    if cfg.llm_provider == "azure_openai":
        missing = [
            name
            for name, value in [
                ("AZURE_OPENAI_LLM_ENDPOINT", cfg.azure_openai_llm_endpoint),
                ("AZURE_OPENAI_LLM_API_VERSION", cfg.azure_openai_llm_api_version),
                ("AZURE_OPENAI_LLM_DEPLOYMENT", cfg.azure_openai_llm_deployment),
                ("AZURE_OPENAI_LLM_API_KEY", cfg.azure_openai_llm_api_key),
            ]
            if not value
        ]
        if missing:
            raise LLMConfigError(
                "Azure OpenAI LLM requires environment variables: " + ", ".join(missing)
            )
        return AzureChatOpenAI(
            deployment_name=cfg.azure_openai_llm_deployment,  # type: ignore[arg-type]
            azure_endpoint=cfg.azure_openai_llm_endpoint,
            openai_api_version=cfg.azure_openai_llm_api_version,
            openai_api_key=cfg.azure_openai_llm_api_key,
            temperature=0,
        )

    if cfg.llm_provider == "openai":
        missing = [
            name
            for name, value in [
                ("OPENAI_BASE_URL", cfg.openai_api_base),
                ("OPENAI_API_KEY", cfg.openai_api_key),
                ("OPENAI_MODEL", cfg.openai_model),
            ]
            if not value
        ]
        if missing:
            raise LLMConfigError(
                "OpenAI-compatible LLM requires environment variables: " + ", ".join(missing)
            )
        return ChatOpenAI(
            base_url=cfg.openai_api_base,
            model=cfg.openai_model,
            api_key=cfg.openai_api_key,
            temperature=0,
        )

    raise LLMConfigError(f"Unsupported LLM provider: {cfg.llm_provider}")
