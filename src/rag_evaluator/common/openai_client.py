"""Helpers to build OpenAI-compatible clients from a RAGConfig.

Core RAG implementations talk to OpenAI-compatible endpoints. The provider,
base URL and API key for generation and for embeddings are independent and are
read from the RAGConfig, falling back to global settings when not set. This lets
the platform point generation and embeddings at different OpenAI-compatible
endpoints (OpenAI, OpenRouter, Ollama, vLLM, ...).

Google Vertex AI Gemini is supported via the Vertex OpenAI-compatible endpoint
(``vertex_ai`` provider). Auth uses Application Default Credentials (ADC): a
short-lived access token is fetched on demand and injected as a bearer header
by :class:`~rag_evaluator.common.gcp_token_provider.GCPBearerAuth`.
"""

from __future__ import annotations

import httpx
from openai import OpenAI

from rag_evaluator.common.base_rag import RAGConfig
from rag_evaluator.config import settings

# Provider identifiers recognized as "Vertex AI Gemini via OpenAI-compat endpoint".
VERTEX_AI_PROVIDERS = frozenset({"vertex_ai", "gemini", "google_vertex_ai"})

# Placeholder api_key value used when the real credential comes from an httpx
# Auth hook (Vertex AI ADC bearer token). The OpenAI SDK requires a non-empty
# api_key; the Auth hook overwrites the Authorization header on every request.
_ADC_API_KEY_SENTINEL = "adc-token-managed-by-gcp-auth"


def is_vertex_ai_provider(provider: str | None) -> bool:
    """Return True if ``provider`` names the Vertex AI OpenAI-compat integration."""
    if not provider:
        return False
    return provider.lower() in VERTEX_AI_PROVIDERS


def _make_vertex_ai_client(timeout: int) -> OpenAI:
    """Build an OpenAI client pointed at the Vertex AI OpenAI-compat endpoint."""
    # Late imports keep google-auth optional (installed via the 'google-vertex' extra).
    from rag_evaluator.common.gcp_token_provider import (
        GCPBearerAuth,
        build_vertex_openai_base_url,
    )

    project = settings.google_cloud_project or settings.google_vertex_project_id
    location = settings.google_cloud_location
    base_url = build_vertex_openai_base_url(project or "", location)
    http_client = httpx.Client(auth=GCPBearerAuth(), timeout=timeout)
    return OpenAI(
        api_key=_ADC_API_KEY_SENTINEL,
        base_url=base_url,
        http_client=http_client,
        timeout=timeout,
    )


def make_client(
    api_key: str | None,
    base_url: str | None,
    timeout: int,
    provider: str | None = None,
) -> OpenAI:
    """Build an OpenAI client for an OpenAI-compatible endpoint.

    When ``provider`` names the Vertex AI integration, an ADC-authenticated
    client pointed at the Vertex OpenAI-compat endpoint is returned; ``api_key``
    and ``base_url`` are ignored in that case.
    """
    if is_vertex_ai_provider(provider):
        return _make_vertex_ai_client(timeout)
    return OpenAI(
        api_key=api_key or settings.openai_api_key,
        base_url=base_url or settings.openai_base_url,
        timeout=timeout,
    )


def llm_client(config: RAGConfig) -> OpenAI:
    """Build the client used for generation/orchestration calls."""
    return make_client(
        config.llm_api_key,
        config.llm_base_url,
        settings.openai_timeout,
        provider=config.llm_provider,
    )


def embedding_client(config: RAGConfig) -> OpenAI:
    """Build the client used for embedding calls (independent endpoint)."""
    return make_client(
        config.embedding_api_key,
        config.embedding_base_url,
        settings.openai_timeout,
        provider=config.embedding_provider or config.llm_provider,
    )


def resolve_llm_model(config: RAGConfig, model: str | None = None) -> str:
    """Return the fully-qualified model name for the LLM call.

    Vertex AI OpenAI-compat requires publisher-prefixed model IDs
    (e.g. ``google/gemini-2.5-flash``); other providers pass through unchanged.
    Falls back to ``config.llm_model`` when ``model`` is not given.
    """
    resolved = model or config.llm_model
    if is_vertex_ai_provider(config.llm_provider):
        from rag_evaluator.common.gcp_token_provider import prepend_google_prefix

        return prepend_google_prefix(resolved)
    return resolved


def resolve_embedding_model(config: RAGConfig, model: str | None = None) -> str:
    """Return the fully-qualified embedding model name (see :func:`resolve_llm_model`)."""
    resolved = model or config.embedding_model
    provider = config.embedding_provider or config.llm_provider
    if is_vertex_ai_provider(provider):
        from rag_evaluator.common.gcp_token_provider import prepend_google_prefix

        return prepend_google_prefix(resolved)
    return resolved


def _openai_kwargs(api_key: str | None, base_url: str | None) -> dict[str, str]:
    """Build kwargs for libraries that construct their own OpenAI client.

    Only includes keys that resolve to a value, so unset entries fall back to
    the openai client defaults (e.g. OPENAI_API_KEY env var).
    """
    kwargs: dict[str, str] = {}
    resolved_key = api_key or settings.openai_api_key
    resolved_url = base_url or settings.openai_base_url
    if resolved_key:
        kwargs["api_key"] = resolved_key
    if resolved_url:
        kwargs["base_url"] = resolved_url
    return kwargs


def _vertex_ai_kwargs() -> dict[str, str]:
    """Build OpenAI-client kwargs for the Vertex AI OpenAI-compat endpoint.

    Used by third-party libraries (e.g. ``neo4j-graphrag``) that construct
    their own ``OpenAI`` client from a plain kwargs dict. The bearer token is
    fetched eagerly here — callers that keep the client alive across the
    ~1h token lifetime should rebuild kwargs periodically.
    """
    from rag_evaluator.common.gcp_token_provider import (
        GCPTokenProvider,
        build_vertex_openai_base_url,
    )

    project = settings.google_cloud_project or settings.google_vertex_project_id
    location = settings.google_cloud_location
    return {
        "api_key": GCPTokenProvider.instance().get_token(),
        "base_url": build_vertex_openai_base_url(project or "", location),
    }


def llm_openai_kwargs(config: RAGConfig) -> dict[str, str]:
    """OpenAI client kwargs for generation (neo4j-graphrag OpenAILLM)."""
    if is_vertex_ai_provider(config.llm_provider):
        return _vertex_ai_kwargs()
    return _openai_kwargs(config.llm_api_key, config.llm_base_url)


def embedding_openai_kwargs(config: RAGConfig) -> dict[str, str]:
    """OpenAI client kwargs for embeddings (neo4j-graphrag OpenAIEmbeddings)."""
    provider = config.embedding_provider or config.llm_provider
    if is_vertex_ai_provider(provider):
        return _vertex_ai_kwargs()
    return _openai_kwargs(config.embedding_api_key, config.embedding_base_url)
