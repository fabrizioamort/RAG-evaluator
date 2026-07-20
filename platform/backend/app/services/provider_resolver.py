"""Resolve LLM providers to OpenAI-compatible endpoint credentials.

The available providers/models are defined in code (see ``rag_registry``). This
module turns a provider name into the concrete ``base_url`` + ``api_key`` used to
build OpenAI-compatible clients in the core RAG implementations. API keys come
from backend settings and are never persisted into config snapshots.

Google Vertex AI (Gemini) uses ADC — no api_key is set; instead
``vertex_project``/``vertex_location`` are populated and consumed by
``LLMProviderService`` when calling LiteLLM's native ``vertex_ai/*`` path.
"""

from dataclasses import dataclass

from app.config import settings

VERTEX_AI_PROVIDER_ALIASES = frozenset({"vertex_ai", "gemini", "google_vertex_ai"})


def is_vertex_ai_provider(provider: str | None) -> bool:
    """Return True if ``provider`` names the Vertex AI Gemini integration."""
    return bool(provider) and provider.lower() in VERTEX_AI_PROVIDER_ALIASES


@dataclass(frozen=True)
class ProviderEndpoint:
    """Resolved OpenAI-compatible endpoint for a provider.

    For OpenAI-compatible providers (openai, openrouter, ollama), ``base_url`` +
    ``api_key`` are populated. For Vertex AI Gemini, ADC handles auth and only
    ``vertex_project``/``vertex_location`` are set.
    """

    base_url: str | None
    api_key: str | None
    vertex_project: str | None = None
    vertex_location: str | None = None


def resolve_provider_endpoint(
    provider: str | None,
    base_url_override: str | None = None,
) -> ProviderEndpoint:
    """Resolve a provider name to its endpoint, honoring an explicit base_url.

    Args:
        provider: Provider identifier (openai, openrouter, ollama, anthropic,
            vertex_ai).
        base_url_override: Explicit base URL from the config; takes precedence
            over the provider default when set.
    """
    name = (provider or "openai").lower()

    if name == "openrouter":
        default_url: str | None = "https://openrouter.ai/api/v1"
        # Fall back to OPENAI_API_KEY: OpenRouter keys are often stored there
        # together with an OpenRouter base_url (OpenAI-compatible usage).
        api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    elif name == "ollama":
        default_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"
        api_key = "ollama"  # OpenAI client requires a non-empty key
    elif name == "anthropic":
        # Not OpenAI-compatible for the core; usable by the LiteLLM judge.
        default_url = None
        api_key = settings.ANTHROPIC_API_KEY
    elif is_vertex_ai_provider(name):
        # Vertex AI Gemini: ADC-based auth via LiteLLM's native vertex_ai/* path.
        return ProviderEndpoint(
            base_url=base_url_override,
            api_key=None,
            vertex_project=settings.GOOGLE_CLOUD_PROJECT,
            vertex_location=settings.GOOGLE_CLOUD_LOCATION,
        )
    else:  # openai and any unknown provider fall back to OpenAI
        default_url = None
        api_key = settings.OPENAI_API_KEY

    return ProviderEndpoint(base_url=base_url_override or default_url, api_key=api_key)


def normalize_model_for_provider(provider: str | None, model: str) -> str:
    """Normalize stored UI model IDs to the model ID expected by the provider API."""
    if (provider or "").lower() == "openrouter":
        return model.removeprefix("openrouter/")
    return model
