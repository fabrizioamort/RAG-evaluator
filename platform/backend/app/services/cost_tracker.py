"""Cost tracking service for LLM operations."""

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from app.utils.logging_config import get_logger
from app.utils.pricing_defaults import get_embedding_pricing, get_model_pricing

logger = get_logger(__name__)


class CostTracker:
    """Service to calculate and aggregate costs for LLM operations."""

    @staticmethod
    def calculate_cost(
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Decimal:
        """Calculate USD cost for a generation operation.

        Args:
            model_name: Name of the LLM model used.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            Calculated cost as a Decimal.
        """
        pricing = get_model_pricing(model_name)

        # Prices are per 1M tokens
        prompt_cost = (Decimal(prompt_tokens) * pricing["prompt"]) / Decimal("1000000")
        completion_cost = (Decimal(completion_tokens) * pricing["completion"]) / Decimal("1000000")

        total_cost = (prompt_cost + completion_cost).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )

        return total_cost

    @staticmethod
    def calculate_embedding_cost(
        model_name: str,
        tokens: int,
    ) -> Decimal:
        """Calculate USD cost for an embedding operation.

        Args:
            model_name: Name of the embedding model used.
            tokens: Number of tokens embedded.

        Returns:
            Calculated cost as a Decimal.
        """
        pricing = get_embedding_pricing(model_name)

        # Prices are per 1M tokens
        total_cost = (Decimal(tokens) * pricing) / Decimal("1000000")

        return total_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# Singleton instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the CostTracker singleton."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
