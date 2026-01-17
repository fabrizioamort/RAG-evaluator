"""Default pricing for LLM models."""

from decimal import Decimal

# Pricing per 1,000,000 tokens in USD
# Based on current OpenAI pricing (as of late 2023/early 2024)
DEFAULT_PRICING = {
    "gpt-5,1": {
        "prompt": Decimal("1.25"),
        "completion": Decimal("10.00"),
    },
    "gpt-5-mini": {
        "prompt": Decimal("0.25"),
        "completion": Decimal("2.00"),
    },
    "gpt-5-nano": {
        "prompt": Decimal("0.05"),
        "completion": Decimal("0.40"),
    },
    "gpt-4o": {
        "prompt": Decimal("5.00"),
        "completion": Decimal("15.00"),
    },
    "gpt-4o-mini": {
        "prompt": Decimal("0.15"),
        "completion": Decimal("0.60"),
    },
    "gpt-4": {
        "prompt": Decimal("30.00"),
        "completion": Decimal("60.00"),
    },
    "gpt-4-turbo": {
        "prompt": Decimal("10.00"),
        "completion": Decimal("30.00"),
    },
    "gpt-3.5-turbo": {
        "prompt": Decimal("0.50"),
        "completion": Decimal("1.50"),
    },
}

# Embedding pricing per 1,000,000 tokens
DEFAULT_EMBEDDING_PRICING = {
    "text-embedding-3-small": Decimal("0.02"),
    "text-embedding-3-large": Decimal("0.13"),
    "text-embedding-ada-002": Decimal("0.10"),
}


def get_model_pricing(model_name: str) -> dict[str, Decimal]:
    """Get prompt and completion pricing for a model.

    Falls back to gpt-4o-mini pricing if model is unknown.
    """
    # Try exact match
    if model_name in DEFAULT_PRICING:
        return DEFAULT_PRICING[model_name]

    # Try prefix match (e.g., gpt-4o-2024-05-13)
    for model, prices in DEFAULT_PRICING.items():
        if model_name.startswith(model):
            return prices

    # Default fallback
    return DEFAULT_PRICING["gpt-4o-mini"]


def get_embedding_pricing(model_name: str) -> Decimal:
    """Get embedding pricing for a model."""
    if model_name in DEFAULT_EMBEDDING_PRICING:
        return DEFAULT_EMBEDDING_PRICING[model_name]

    # Default fallback
    return DEFAULT_EMBEDDING_PRICING["text-embedding-3-small"]
