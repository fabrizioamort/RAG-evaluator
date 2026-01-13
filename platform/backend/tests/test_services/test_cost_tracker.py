"""Unit tests for CostTracker service."""

from decimal import Decimal

from app.services.cost_tracker import get_cost_tracker


def test_calculate_cost_gpt4o_mini() -> None:
    tracker = get_cost_tracker()
    # gpt-4o-mini: $0.15 / 1M prompt, $0.60 / 1M completion
    # 100,000 prompt tokens = $0.015
    # 50,000 completion tokens = $0.03
    # Total = $0.045
    cost = tracker.calculate_cost("gpt-4o-mini", 100000, 50000)
    assert cost == Decimal("0.045000")


def test_calculate_cost_gpt4o() -> None:
    tracker = get_cost_tracker()
    # gpt-4o: $5.00 / 1M prompt, $15.00 / 1M completion
    # 1,000 prompt tokens = $0.005
    # 500 completion tokens = $0.0075
    # Total = $0.0125
    cost = tracker.calculate_cost("gpt-4o", 1000, 500)
    assert cost == Decimal("0.012500")


def test_calculate_cost_unknown_model_fallback() -> None:
    tracker = get_cost_tracker()
    # Should fallback to gpt-4o-mini
    cost_unknown = tracker.calculate_cost("unknown-model", 1000, 1000)
    cost_mini = tracker.calculate_cost("gpt-4o-mini", 1000, 1000)
    assert cost_unknown == cost_mini


def test_calculate_embedding_cost() -> None:
    tracker = get_cost_tracker()
    # text-embedding-3-small: $0.02 / 1M tokens
    # 1,000,000 tokens = $0.02
    cost = tracker.calculate_embedding_cost("text-embedding-3-small", 1000000)
    assert cost == Decimal("0.020000")


def test_calculate_cost_rounding() -> None:
    tracker = get_cost_tracker()
    # Test very small costs
    cost = tracker.calculate_cost("gpt-4o-mini", 1, 1)
    # 0.15/1M + 0.60/1M = 0.75/1M = 0.00000075
    # Rounded to 6 decimal places: 0.000001
    assert cost == Decimal("0.000001")
