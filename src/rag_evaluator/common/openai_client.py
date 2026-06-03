"""Helpers to build OpenAI-compatible clients from a RAGConfig.

Core RAG implementations talk to OpenAI-compatible endpoints. The provider,
base URL and API key for generation and for embeddings are independent and are
read from the RAGConfig, falling back to global settings when not set. This lets
the platform point generation and embeddings at different OpenAI-compatible
endpoints (OpenAI, OpenRouter, Ollama, vLLM, ...).
"""

from openai import OpenAI

from rag_evaluator.common.base_rag import RAGConfig
from rag_evaluator.config import settings


def make_client(api_key: str | None, base_url: str | None, timeout: int) -> OpenAI:
    """Build an OpenAI client for an OpenAI-compatible endpoint."""
    return OpenAI(
        api_key=api_key or settings.openai_api_key,
        base_url=base_url or settings.openai_base_url,
        timeout=timeout,
    )


def llm_client(config: RAGConfig) -> OpenAI:
    """Build the client used for generation/orchestration calls."""
    return make_client(config.llm_api_key, config.llm_base_url, settings.openai_timeout)


def embedding_client(config: RAGConfig) -> OpenAI:
    """Build the client used for embedding calls (independent endpoint)."""
    return make_client(
        config.embedding_api_key,
        config.embedding_base_url,
        settings.openai_timeout,
    )


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


def llm_openai_kwargs(config: RAGConfig) -> dict[str, str]:
    """OpenAI client kwargs for generation (neo4j-graphrag OpenAILLM)."""
    return _openai_kwargs(config.llm_api_key, config.llm_base_url)


def embedding_openai_kwargs(config: RAGConfig) -> dict[str, str]:
    """OpenAI client kwargs for embeddings (neo4j-graphrag OpenAIEmbeddings)."""
    return _openai_kwargs(config.embedding_api_key, config.embedding_base_url)
