"""
Upload validation service.

Separating validation from the API endpoint handler keeps the route thin
and makes each validation rule independently unit-testable. Adding a new
rule (e.g. minimum file duration) requires only a new method here, not
changes to the route or any other layer.

Validation order matters for user experience:
  1. Presence check  → 400 (no point reading headers if file is missing)
  2. Extension check → 415 (cheap string op before touching file content)
  3. MIME type check → 415 (python-magic reads only file header bytes)
  4. Size check      → 413 (avoids streaming the whole file unnecessarily)
"""

from pathlib import Path

import magic  # python-magic: wraps libmagic for accurate MIME detection

from app.config.settings import Settings
from app.utils.exceptions import FileMissingError, FileTooLargeError, UnsupportedFormatError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class UploadValidator:
    """
    Validates uploaded audio files against a configured allow-list.

    This class is intentionally stateless beyond its settings reference,
    making it thread-safe and cheap to instantiate per request if needed.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Application settings carrying allowed extensions,
                      MIME types, and max upload size.
        """
        self._settings = settings

    def validate(self, filename: str | None, file_path: Path, file_size: int) -> None:
        """
        Run all validation checks on an uploaded file.

        Raises the first validation error encountered (fast-fail strategy)
        rather than collecting all errors, because the most common case is
        a valid file and we want to minimize overhead on the happy path.

        Args:
            filename: Original filename reported by the client (may be None
                      if the multipart field has no filename).
            file_path: Local path where the upload was saved.
            file_size: Size of the uploaded file in bytes.

        Raises:
            FileMissingError: If filename is absent.
            UnsupportedFormatError: If extension or MIME type is not allowed.
            FileTooLargeError: If size exceeds the configured maximum.
        """
        self._check_filename_present(filename)
        self._check_extension(filename)  # type: ignore[arg-type]  # guarded above
        self._check_mime_type(filename, file_path)  # type: ignore[arg-type]
        self._check_size(file_size)

        logger.info(
            "File validation passed | filename=%s size_bytes=%d",
            filename,
            file_size,
        )

    # ------------------------------------------------------------------ #
    # Private validation steps
    # ------------------------------------------------------------------ #

    def _check_filename_present(self, filename: str | None) -> None:
        """Reject requests with no filename (empty or missing form field)."""
        if not filename or not filename.strip():
            raise FileMissingError()

    def _check_extension(self, filename: str) -> None:
        """
        Reject files whose extension is not in the allow-list.

        Extension checks are a first line of defence and a UX signal —
        they catch obvious misuse (e.g. uploading a PDF) before wasting
        cycles on MIME detection.
        """
        extension = Path(filename).suffix.lstrip(".").lower()
        if extension not in self._settings.allowed_extensions:
            # We do not yet know the MIME type, so pass an informative placeholder.
            raise UnsupportedFormatError(
                filename=filename,
                mime_type="(not checked)",
                allowed=self._settings.allowed_extensions,
            )

    def _check_mime_type(self, filename: str, file_path: Path) -> None:
        """
        Validate the actual MIME type via libmagic header inspection.

        python-magic reads the first few hundred bytes of the file and
        compares them against a database of magic byte signatures. This
        prevents extension spoofing: a JPEG renamed to 'audio.wav' will
        be detected as 'image/jpeg' and rejected.

        We fall back gracefully if libmagic is unavailable, logging a
        warning rather than crashing — extension validation already provides
        a baseline level of protection.
        """
        try:
            detected_mime = magic.from_file(str(file_path), mime=True)
        except Exception as exc:
            # Non-fatal: libmagic might be missing on some platforms.
            # Log and skip rather than blocking valid uploads.
            logger.warning(
                "MIME type detection failed, skipping check | filename=%s error=%s",
                filename,
                exc,
            )
            return

        if detected_mime not in self._settings.allowed_mime_types:
            raise UnsupportedFormatError(
                filename=filename,
                mime_type=detected_mime,
                allowed=self._settings.allowed_extensions,
            )

        logger.debug("MIME type validated | filename=%s mime=%s", filename, detected_mime)

    def _check_size(self, file_size: int) -> None:
        """
        Reject files that exceed the configured maximum size.

        We check size after saving to disk (not by reading Content-Length
        headers) because Content-Length can be spoofed by clients. The
        actual byte count read from the stream is authoritative.
        """
        max_bytes = self._settings.max_upload_size_bytes
        if file_size > max_bytes:
            raise FileTooLargeError(size_bytes=file_size, max_bytes=max_bytes)
