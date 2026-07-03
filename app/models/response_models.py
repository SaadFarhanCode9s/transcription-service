"""
Pydantic response models for the transcription API.

Using Pydantic models for responses (not just inputs) provides:
1. Automatic JSON serialization with consistent field names.
2. OpenAPI schema generation — FastAPI renders these in /docs automatically.
3. Validation of our own output, catching bugs where internal types
   do not match the documented contract.
4. Easy versioning: add Optional fields without breaking existing clients.

We use strict type annotations throughout. Optional fields use Python 3.10+
union syntax (X | None) for clarity.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentModel(BaseModel):
    """
    A single transcribed segment (sentence or phrase) with timing information.

    Segments come directly from Whisper's internal attention mechanism —
    they represent natural speech boundaries rather than fixed-length windows.
    """

    start: float = Field(..., description="Segment start time in seconds from the beginning of the audio", ge=0.0)
    end: float = Field(..., description="Segment end time in seconds from the beginning of the audio", ge=0.0)
    text: str = Field(..., description="Transcribed text for this segment")

    model_config = {"json_schema_extra": {"example": {"start": 0.0, "end": 3.25, "text": "Hello everyone."}}}


class TranscriptionResponse(BaseModel):
    """
    Top-level response returned by POST /transcribe.

    The `segments` list is always present (never null) — for very short
    audio that Whisper processes as one chunk, it will contain a single entry.
    """

    language: str = Field(
        ...,
        description="ISO 639-1 language code detected by Whisper (e.g. 'en', 'fr', 'de'). "
                    "Forced to the configured language if WHISPER_LANGUAGE is set.",
    )
    duration: float = Field(
        ...,
        description="Total audio duration in seconds",
        ge=0.0,
    )
    transcription: str = Field(
        ...,
        description="Full concatenated transcription text (all segments joined with spaces)",
    )
    segments: list[SegmentModel] = Field(
        default_factory=list,
        description="List of timed transcription segments",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "language": "en",
                "duration": 31.5,
                "transcription": "Hello everyone. Welcome to the demonstration.",
                "segments": [
                    {"start": 0.0, "end": 3.25, "text": "Hello everyone."},
                    {"start": 3.5, "end": 7.0, "text": "Welcome to the demonstration."},
                ],
            }
        }
    }


class HealthResponse(BaseModel):
    """
    Response for GET /health.

    Kubernetes liveness and readiness probes call this endpoint.
    Returning structured JSON (not plain text) lets monitoring tools
    parse service state without string matching.
    """

    status: str = Field(..., description="Service health status: 'ok' | 'degraded' | 'error'")
    version: str = Field(..., description="Application version")
    whisper_model: str = Field(..., description="Loaded Whisper model identifier")
    ffmpeg_available: bool = Field(..., description="Whether FFmpeg binary is present and executable")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "whisper_model": "base",
                "ffmpeg_available": True,
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Standardized error envelope returned for all 4xx and 5xx responses.

    Using a consistent error schema means clients need only one error-parsing
    code path regardless of which endpoint produced the error.
    """

    error: str = Field(..., description="Machine-readable snake_case error code")
    message: str = Field(..., description="Human-readable error description")
    details: dict = Field(default_factory=dict, description="Optional structured context about the error")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "unsupported_format",
                "message": "Unsupported file format 'audio.xyz'. Accepted formats: wav, mp3, flac, m4a, ogg, aac.",
                "details": {"filename": "audio.xyz", "mime_type": "application/octet-stream"},
            }
        }
    }
