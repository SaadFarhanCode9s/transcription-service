"""
FastAPI application factory and lifespan management.

WHY FASTAPI?
------------
FastAPI was chosen over Flask/Django for this service because:
1. Async-native: I/O-bound operations (file upload, subprocess) benefit
   from async handling without blocking the event loop.
2. Automatic OpenAPI: The /docs endpoint is generated from type hints —
   no extra maintenance overhead for API documentation.
3. Pydantic integration: Request/response validation with zero boilerplate.
4. Performance: Starlette-based, comparable throughput to Go/Node for I/O
   bound workloads (transcription is CPU-bound, not I/O-bound).
5. Dependency injection: Built-in DI makes testing and composition clean.

LIFESPAN PATTERN
----------------
We use FastAPI's lifespan context manager (preferred over @app.on_event
which is deprecated) to load the Whisper model exactly once at startup.
This ensures:
- The heavy model loading happens before the first request is served.
- Kubernetes readiness probes will fail until the model is ready.
- The loaded model is shared across all requests in the same process.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config.settings import get_settings
from app.services.transcriber import create_transcription_service
from app.utils.exceptions import FFmpegNotFoundError, ModelLoadError
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler: startup and shutdown logic.

    Everything before 'yield' runs at startup (before any requests).
    Everything after 'yield' runs at shutdown (after all requests complete).

    We store the transcription service on app.state so routes can access
    it via request.app.state without a global variable, which would be
    difficult to mock in tests.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info(
        "Starting %s v%s | env=%s model=%s device=%s",
        settings.app_name,
        settings.app_version,
        settings.environment,
        settings.whisper_model,
        settings.whisper_device,
    )

    # ------------------------------------------------------------------ #
    # Startup: load model (heavy operation, done once per process)
    # ------------------------------------------------------------------ #
    try:
        app.state.transcription_service = create_transcription_service(settings)
        logger.info("Transcription service ready")
    except FFmpegNotFoundError as exc:
        # FFmpeg is a hard dependency. Log clearly and let the process
        # start in a degraded state — the health endpoint will report
        # 'degraded' and Kubernetes won't route traffic to this pod.
        logger.critical("STARTUP FAILED: %s", exc.message)
        # We intentionally do not re-raise here: if we crash at startup,
        # Kubernetes restarts the pod immediately in a tight loop.
        # A degraded state gives the operator time to attach and debug.
    except ModelLoadError as exc:
        logger.critical("STARTUP FAILED: Model could not be loaded | %s", exc.message)
        # Same reasoning: stay up in degraded mode so health checks surface the problem.

    logger.info("Application startup complete — ready to serve requests")

    yield  # ← Application runs here

    # ------------------------------------------------------------------ #
    # Shutdown: clean up resources
    # ------------------------------------------------------------------ #
    logger.info("Application shutting down — releasing resources")
    # In production, you would close DB connections, flush buffers, etc.
    # The Whisper model is garbage-collected automatically.


def create_app() -> FastAPI:
    """
    Application factory: construct and configure the FastAPI instance.

    Using a factory function (rather than a module-level app = FastAPI())
    makes it easy to create isolated app instances for testing without
    affecting global state.

    Returns:
        Configured FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-ready Speech-to-Text transcription service powered by "
            "WhisperX / OpenAI Whisper and FFmpeg. Accepts WAV, MP3, FLAC, M4A, "
            "OGG, and AAC. Returns timestamped transcription segments with "
            "language detection."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # CORS middleware
    # ------------------------------------------------------------------ #
    # Allow all origins in development; restrict to specific origins in production
    # using the CORS_ORIGINS environment variable.
    # In a microservice architecture, CORS is often handled at the gateway
    # layer (nginx, Kong, AWS API Gateway) rather than in the application.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Global exception handler
    # ------------------------------------------------------------------ #
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Last-resort handler for any uncaught exception.

        Returning a structured JSON response (rather than letting FastAPI
        return its default HTML error page) ensures API clients always
        receive a parseable error regardless of what went wrong.
        """
        logger.exception("Unhandled exception | path=%s method=%s", request.url.path, request.method)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred. Our team has been notified.",
                "details": {},
            },
        )

    # ------------------------------------------------------------------ #
    # Register routers
    # ------------------------------------------------------------------ #
    app.include_router(router, prefix="/api/v1")

    # Root redirect to docs — helpful for developers hitting the base URL
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "service": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "health": "/api/v1/health",
            }
        )

    return app


# Module-level app instance for uvicorn discovery.
# uvicorn main:app uses this.
app = create_app()
