.PHONY: help install test lint format clean ui

help:
	@echo "RAG Evaluator - Available commands:"
	@echo "  make install    - Install all dependencies"
	@echo "  make test       - Run tests with coverage"
	@echo "  make lint       - Run linting and type checking"
	@echo "  make format     - Format code with ruff"
	@echo "  make clean      - Clean generated files"
	@echo "  make ui         - Launch Streamlit UI"
	@echo "  make check      - Run format, lint, and test"

install:
	uv sync --all-extras

test:
	uv run pytest --cov=src/rag_evaluator --cov-report=term-missing

lint:
	uv run ruff check .
	uv run mypy src/rag_evaluator

format:
	uv run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage

ui:
	uv run python scripts/run_streamlit.py

check: format lint test
	@echo "All checks passed!"
