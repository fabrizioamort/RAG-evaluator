.PHONY: help install install-backend install-frontend \
        test test-core test-backend test-frontend \
        lint lint-core lint-backend lint-frontend \
        format clean check check-parallel \
        dev-backend dev-frontend dev-infra

# =============================================================================
# Help
# =============================================================================

help:
	@echo "RAG Evaluator Platform - Available commands:"
	@echo ""
	@echo "  Installation:"
	@echo "    make install          - Install all dependencies (core + backend + frontend)"
	@echo "    make install-backend  - Install backend dependencies only"
	@echo "    make install-frontend - Install frontend dependencies only"
	@echo ""
	@echo "  Testing:"
	@echo "    make test             - Run all tests (core + backend)"
	@echo "    make test-core        - Run core library tests"
	@echo "    make test-backend     - Run backend API tests"
	@echo "    make test-frontend    - Run frontend tests (vitest)"
	@echo ""
	@echo "  Linting:"
	@echo "    make lint             - Run all linters"
	@echo "    make lint-core        - Lint core library (ruff + mypy)"
	@echo "    make lint-backend     - Lint backend (ruff)"
	@echo "    make lint-frontend    - Lint frontend (eslint)"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make format           - Format all Python code with ruff"
	@echo "    make check            - Run all checks (format + lint + test)"
	@echo "    make check-parallel   - Run independent checks in parallel"
	@echo ""
	@echo "  Development:"
	@echo "    make dev-backend      - Start backend server (uvicorn)"
	@echo "    make dev-frontend     - Start frontend dev server (vite)"
	@echo "    make dev-infra        - Start infrastructure (postgres, qdrant, neo4j)"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean            - Clean generated files and caches"

# =============================================================================
# Installation
# =============================================================================

install: install-backend install-frontend
	uv sync --all-extras

install-backend:
	cd platform/backend && uv sync --all-extras

install-frontend:
	cd platform/frontend && npm install

# =============================================================================
# Testing
# =============================================================================

test: test-core test-backend
	@echo "All tests passed!"

test-core:
	uv run pytest --cov=src/rag_evaluator --cov-report=term-missing

test-backend:
	cd platform/backend && uv run pytest

test-frontend:
	cd platform/frontend && npm run test

# =============================================================================
# Linting
# =============================================================================

lint: lint-core lint-backend lint-frontend
	@echo "All linting passed!"

lint-core:
	uv run ruff check .
	uv run mypy src/rag_evaluator

lint-backend:
	cd platform/backend && uv run ruff check .

lint-frontend:
	cd platform/frontend && npm run lint

# =============================================================================
# Formatting
# =============================================================================

format:
	uv run ruff format .
	cd platform/backend && uv run ruff format .

# =============================================================================
# Combined Checks
# =============================================================================

check: format lint test
	@echo "All checks passed!"

# Run independent checks in parallel (requires make -j)
# Usage: make check-parallel -j3
check-parallel: lint-core lint-backend lint-frontend test-core test-backend
	@echo "All parallel checks passed!"

# =============================================================================
# Development Servers
# =============================================================================

dev-backend:
	cd platform/backend && uv run uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd platform/frontend && npm run dev

dev-infra:
	docker-compose up -d postgres qdrant neo4j

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -prune -o -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage
	@echo "Cleaned generated files and caches"
