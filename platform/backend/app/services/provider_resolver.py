"""Resolve LLM providers to OpenAI-compatible endpoint credentials.

The available providers/models are defined in code (see ``rag_registry``). This
module turns a provider name into the concrete ``base_url`` + ``api_key`` used to
build OpenAI-compatible clients in the core RAG implementations. API keys come
from backend settings and are never persisted into config snapshots.
"""

from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class ProviderEndpoint:
    """Resolved OpenAI-compatible endpoint for a provider."""

    base_url: str | None
    api_key: str | None


def resolve_provider_endpoint(
    provider: str | None,
    base_url_override: str | None = None,
) -> ProviderEndpoint:
    """Resolve a provider name to its endpoint, honoring an explicit base_url.

    Args:
        provider: Provider identifier (openai, openrouter, ollama, anthropic).
        base_url_override: Explicit base URL from the config; takes precedence
            over the provider default when set.
    """
    name = (provider or "openai").lower()

    if name == "openrouter":
        default_url: str | None = "https://openrouter.ai/api/v1"
        api_key = settings.OPENROUTER_API_KEY
    elif name == "ollama":
        default_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"
        api_key = "ollama"  # OpenAI client requires a non-empty key
    elif name == "anthropic":
        # Not OpenAI-compatible for the core; usable by the LiteLLM judge.
        default_url = None
        api_key = settings.ANTHROPIC_API_KEY
    else:  # openai and any unknown provider fall back to OpenAI
        default_url = None
        api_key = settings.OPENAI_API_KEY

    return ProviderEndpoint(base_url=base_url_override or default_url, api_key=api_key)
