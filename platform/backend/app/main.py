"""FastAPI application entrypoint."""

import uuid
import warnings

# Suppress pydantic serialization warnings from litellm internal response models
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.api import (
    comparisons,
    evaluations,
    health,
    indexes,
    knowledge_bases,
    playground,
    projects,
    rag_configs,
    test_sets,
    test_templates,
    trends,
    webhooks,
)
from app.config import settings
from app.database import engine, init_db
from app.schemas.errors import ErrorResponse
from app.utils.exceptions import AppException
from app.utils.logging_config import get_logger, request_id_var, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    setup_logging()
    logger.info(
        "Starting application",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )

    # Ensure storage directories exist
    storage_path = Path(settings.STORAGE_PATH)
    for subdir in ["documents", "indexes", "artifacts", "reports"]:
        (storage_path / subdir).mkdir(parents=True, exist_ok=True)

    # Initialize database (create tables if using SQLite in dev)
    if settings.is_sqlite:
        await init_db()
        logger.info("Database initialized (SQLite mode)")

    # Load builtin templates
    from app.database import get_db_context
    from app.utils.template_loader import load_builtin_templates

    async with get_db_context() as db:
        await load_builtin_templates(db)

    yield

    # Shutdown
    logger.info("Shutting down application")
    from app.services.rag_adapter import get_rag_adapter_service
    from app.services.webhook_service import get_webhook_service

    get_rag_adapter_service().clear_cache()
    await get_webhook_service().close()
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="Platform for evaluating RAG (Retrieval Augmented Generation) systems",
    version=settings.APP_VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable[[Request], Any]) -> Response:
    """Add request ID to each request for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(request_id)

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


@app.middleware("http")
async def log_requests_middleware(
    request: Request, call_next: Callable[[Request], Any]
) -> Response:
    """Log incoming requests."""
    logger.info(
        "Request received",
        method=request.method,
        path=request.url.path,
    )

    response: Response = await call_next(request)

    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    )

    return response


# Exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application-specific exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            request_id=request_id_var.get(),
            errors=exc.errors,
        ).model_dump(exclude_none=True),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            detail="Validation failed",
            request_id=request_id_var.get(),
            errors=list(exc.errors()),
        ).model_dump(exclude_none=True),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error",
            request_id=request_id_var.get(),
            errors=None,
        ).model_dump(exclude_none=True),
    )


# Include routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(projects.router, prefix=settings.API_V1_PREFIX)
app.include_router(knowledge_bases.router, prefix=settings.API_V1_PREFIX)
app.include_router(indexes.router, prefix=settings.API_V1_PREFIX)  # Added
app.include_router(test_sets.router, prefix=settings.API_V1_PREFIX)
app.include_router(test_templates.router, prefix=settings.API_V1_PREFIX)
app.include_router(rag_configs.router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluations.router, prefix=settings.API_V1_PREFIX)
app.include_router(comparisons.router, prefix=settings.API_V1_PREFIX)
app.include_router(trends.router, prefix=settings.API_V1_PREFIX)
app.include_router(webhooks.router, prefix=settings.API_V1_PREFIX)
app.include_router(playground.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint redirect info."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": f"{settings.API_V1_PREFIX}/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }
