"""
Transcription API router.

This module contains only route definitions and request/response wiring.
ALL business logic lives in the service layer — the router's sole job is
to translate HTTP concerns (multipart form data, status codes, headers)
into service calls and back.

This strict separation means:
- Services can be tested without spinning up an HTTP server.
- Routes can be versioned (v1 → v2) without touching service code.
- The API layer is thin enough to read and understand in one sitting.
"""

import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config.settings import Settings, get_settings
from app.models.response_models import ErrorResponse, HealthResponse, TranscriptionResponse
from app.services.audio_normalizer import AudioNormalizer
from app.services.storage import LocalFileStorage
from app.services.transcriber import TranscriptionService
from app.services.validator import UploadValidator
from app.utils.exceptions import (
    AudioNormalizationError,
    FileMissingError,
    FileTooLargeError,
    TranscriptionError,
    UnsupportedFormatError,
)
from app.utils.file_helpers import generate_unique_filename
from app.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Dependency helpers
# --------------------------------------------------------------------------- #


def get_validator(settings: Settings = Depends(get_settings)) -> UploadValidator:
    """Provide an UploadValidator scoped to the current request."""
    return UploadValidator(settings)


def get_storage(settings: Settings = Depends(get_settings)) -> LocalFileStorage:
    """Provide a LocalFileStorage scoped to the current request."""
    return LocalFileStorage(settings)


def get_normalizer(settings: Settings = Depends(get_settings)) -> AudioNormalizer:
    """Provide an AudioNormalizer scoped to the current request."""
    return AudioNormalizer(settings)


def get_transcription_service(request: Request) -> TranscriptionService:
    """
    Retrieve the singleton TranscriptionService from application state.

    The model is loaded once at startup (see main.py lifespan) and stored
    in app.state to avoid per-request model loading overhead. FastAPI's
    dependency injection makes this available to every route that needs it.
    """
    service: TranscriptionService = request.app.state.transcription_service
    return service


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe an audio file",
    description=(
        "Upload a WAV, MP3, FLAC, M4A, OGG, or AAC audio file. "
        "The service normalizes the audio using FFmpeg and transcribes it "
        "using Whisper, returning language detection, duration, full text, "
        "and timestamped segments."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Missing file"},
        413: {"model": ErrorResponse, "description": "File exceeds size limit"},
        415: {"model": ErrorResponse, "description": "Unsupported audio format"},
        422: {"model": ErrorResponse, "description": "Audio could not be processed"},
        500: {"model": ErrorResponse, "description": "Internal transcription error"},
    },
    tags=["Transcription"],
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    settings: Settings = Depends(get_settings),
    validator: UploadValidator = Depends(get_validator),
    storage: LocalFileStorage = Depends(get_storage),
    normalizer: AudioNormalizer = Depends(get_normalizer),
    transcription_service: TranscriptionService = Depends(get_transcription_service),
) -> JSONResponse:
    """
    POST /transcribe — Main transcription endpoint.

    Processing pipeline:
    1. Save upload to staging area.
    2. Validate (presence, extension, MIME type, size).
    3. Normalize audio with FFmpeg (16 kHz mono WAV).
    4. Transcribe with WhisperX / OpenAI Whisper.
    5. Persist result JSON.
    6. Return structured response.
    7. Clean up temporary files (always, even on error).

    We stage the file to disk before validation rather than validating
    from the stream because:
    a) MIME type detection requires reading file content (not headers).
    b) Size validation requires knowing the actual byte count streamed.
    c) FFmpeg requires a file path, not a stream.
    """
    request_start = time.perf_counter()
    upload_path: Path | None = None
    normalized_path: Path | None = None

    try:
        # ------------------------------------------------------------------ #
        # Step 1: Stage upload to disk
        # ------------------------------------------------------------------ #
        safe_filename = generate_unique_filename(file.filename or "upload.bin")
        upload_path = storage.get_upload_path(safe_filename)

        logger.info(
            "Receiving upload | original_name=%s staged_path=%s",
            file.filename,
            upload_path,
        )

        file_size = 0
        with open(upload_path, "wb") as dest:
            while chunk := await file.read(65_536):  # 64 KB streaming read
                file_size += len(chunk)
                dest.write(chunk)

        # ------------------------------------------------------------------ #
        # Step 2: Validate
        # ------------------------------------------------------------------ #
        validator.validate(
            filename=file.filename,
            file_path=upload_path,
            file_size=file_size,
        )

        # ------------------------------------------------------------------ #
        # Step 3: Normalize audio
        # ------------------------------------------------------------------ #
        with log_execution_time(logger, "audio_normalization"):
            normalized_path = normalizer.normalize(
                input_path=upload_path,
                output_dir=upload_path.parent,
            )

        # ------------------------------------------------------------------ #
        # Step 4: Transcribe
        # ------------------------------------------------------------------ #
        with log_execution_time(logger, "transcription"):
            result = transcription_service.transcribe(normalized_path)

        # ------------------------------------------------------------------ #
        # Step 5: Persist result
        # ------------------------------------------------------------------ #
        result_path = storage.save_result(stem=normalized_path.stem, result=result)
        logger.info("Result persisted | path=%s", result_path)

        # ------------------------------------------------------------------ #
        # Step 6: Return response
        # ------------------------------------------------------------------ #
        elapsed = time.perf_counter() - request_start
        logger.info(
            "Transcription complete | language=%s duration=%.2fs segments=%d elapsed=%.2fs",
            result["language"],
            result["duration"],
            len(result["segments"]),
            elapsed,
        )

        return JSONResponse(content=result, status_code=status.HTTP_200_OK)

    # ------------------------------------------------------------------ #
    # Error handling: map domain exceptions to HTTP responses
    # ------------------------------------------------------------------ #
    except FileMissingError as exc:
        logger.warning("Upload rejected: file missing | %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    except UnsupportedFormatError as exc:
        logger.warning("Upload rejected: unsupported format | filename=%s mime=%s", file.filename, exc.details.get("mime_type"))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    except FileTooLargeError as exc:
        logger.warning("Upload rejected: file too large | size_bytes=%d", exc.details.get("size_bytes"))
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    except AudioNormalizationError as exc:
        logger.error("Audio normalization failed | %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    except TranscriptionError as exc:
        logger.error("Transcription failed | %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    except Exception as exc:
        # Catch-all: log the full traceback for debugging but return a
        # generic message to the client to avoid leaking implementation details.
        logger.exception("Unhandled exception during transcription | error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred. Please try again.",
                "details": {},
            },
        )

    finally:
        # Step 7: Always clean up — regardless of success or failure.
        # Files left on disk accumulate silently and exhaust disk space.
        paths_to_clean = [p for p in (upload_path, normalized_path) if p is not None]
        storage.cleanup_upload(*paths_to_clean)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description=(
        "Returns service health status. Used by Kubernetes liveness and "
        "readiness probes, load balancers, and monitoring systems."
    ),
    tags=["Operations"],
)
async def health_check(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """
    GET /health — Health and readiness endpoint.

    We check FFmpeg availability dynamically (not cached) because an operator
    might remove the binary after startup. A stale cached 'true' would mask
    a broken environment.

    Returns HTTP 200 with 'ok' status when fully operational,
    or HTTP 503 with 'degraded' if a dependency is unavailable.
    """
    import shutil

    ffmpeg_ok = shutil.which("ffmpeg") is not None

    # Check if the transcription service loaded successfully.
    service_ok = hasattr(request.app.state, "transcription_service")

    overall_status = "ok" if (ffmpeg_ok and service_ok) else "degraded"
    http_status = status.HTTP_200_OK if overall_status == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE

    response_body = {
        "status": overall_status,
        "version": settings.app_version,
        "whisper_model": settings.whisper_model,
        "ffmpeg_available": ffmpeg_ok,
    }

    logger.debug("Health check | status=%s ffmpeg=%s", overall_status, ffmpeg_ok)
    return JSONResponse(content=response_body, status_code=http_status)
