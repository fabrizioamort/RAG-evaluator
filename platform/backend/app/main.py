"""FastAPI application entrypoint."""

import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.api import health
from app.config import settings
from app.database import engine, init_db
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

    yield

    # Shutdown
    logger.info("Shutting down application")
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
        content={
            "detail": "Internal server error",
            "request_id": request_id_var.get(),
        },
    )


# Include routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint redirect info."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": f"{settings.API_V1_PREFIX}/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }
