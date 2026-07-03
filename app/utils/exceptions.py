"""
Custom exception hierarchy for the transcription service.

Having a dedicated exception tree serves several purposes:
1. API layer can catch specific exceptions and return appropriate HTTP codes
   without inspecting error messages (string matching is fragile).
2. Middleware can log structured error context without duplication.
3. Each exception carries a machine-readable `code` field so client code
   can switch on the error type without parsing human-facing messages.

All service exceptions inherit from TranscriptionServiceError so callers
can catch the entire family with a single except clause when needed.
"""


class TranscriptionServiceError(Exception):
    """
    Base class for all application-level errors.

    Attributes:
        message: Human-readable description of the error.
        code: Machine-readable snake_case error code for client switching.
        details: Optional dictionary with additional structured context
                 (e.g. {"filename": "...", "size_bytes": 123}).
    """

    def __init__(self, message: str, code: str = "service_error", details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------------------- #
# Validation errors (HTTP 400 / 413 / 415)
# --------------------------------------------------------------------------- #


class FileMissingError(TranscriptionServiceError):
    """Raised when no file is included in the multipart/form-data request."""

    def __init__(self) -> None:
        super().__init__(
            message="No audio file was included in the request. "
                    "Send the file under the 'file' form-data field.",
            code="file_missing",
        )


class UnsupportedFormatError(TranscriptionServiceError):
    """
    Raised when the uploaded file has an extension or MIME type that is
    not in the allow-list.

    We validate both extension AND MIME type to prevent trivial bypass
    attacks where an attacker renames e.g. an executable to .wav.
    """

    def __init__(self, filename: str, mime_type: str, allowed: list[str]) -> None:
        super().__init__(
            message=(
                f"Unsupported file format '{filename}' (MIME: {mime_type}). "
                f"Accepted formats: {', '.join(allowed)}."
            ),
            code="unsupported_format",
            details={"filename": filename, "mime_type": mime_type, "allowed_extensions": allowed},
        )


class FileTooLargeError(TranscriptionServiceError):
    """
    Raised when the upload exceeds the configured maximum size.

    Enforcing a size limit at the application layer prevents runaway
    memory usage during normalization. The limit should also be enforced
    at the reverse proxy level (nginx client_max_body_size).
    """

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        super().__init__(
            message=(
                f"File size {size_bytes / 1_048_576:.1f} MB exceeds the "
                f"maximum allowed size of {max_bytes / 1_048_576:.0f} MB."
            ),
            code="file_too_large",
            details={"size_bytes": size_bytes, "max_bytes": max_bytes},
        )


# --------------------------------------------------------------------------- #
# Processing errors (HTTP 422 / 500)
# --------------------------------------------------------------------------- #


class AudioNormalizationError(TranscriptionServiceError):
    """
    Raised when FFmpeg fails to normalize the audio file.

    This is a 422 Unprocessable Entity scenario: the file passed validation
    checks (correct extension, MIME type, size) but FFmpeg cannot decode it,
    meaning the file is corrupt or malformed despite appearing valid.
    """

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to normalize audio file '{filename}': {reason}",
            code="audio_normalization_error",
            details={"filename": filename, "reason": reason},
        )


class TranscriptionError(TranscriptionServiceError):
    """
    Raised when the Whisper model fails to transcribe the audio.

    This is a 500 Internal Server Error — the input was valid, but inference
    failed due to a model or infrastructure issue.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Transcription failed: {reason}",
            code="transcription_error",
            details={"reason": reason},
        )


class FFmpegNotFoundError(TranscriptionServiceError):
    """
    Raised at startup if the ffmpeg binary is not available on PATH.

    Failing fast at startup (rather than on the first request) gives
    operators a clear signal that a runtime dependency is missing.
    """

    def __init__(self) -> None:
        super().__init__(
            message=(
                "ffmpeg binary not found on PATH. "
                "Install ffmpeg: https://ffmpeg.org/download.html"
            ),
            code="ffmpeg_not_found",
        )


class ModelLoadError(TranscriptionServiceError):
    """Raised when the Whisper/WhisperX model fails to load."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            code="model_load_error",
            details={"model_name": model_name, "reason": reason},
        )
