"""
Storage abstraction for managing upload and output file lifecycle.

Separating storage concerns from business logic serves two goals:
1. Testability: tests can inject an in-memory or temp-dir storage
   without needing to mock filesystem calls scattered across services.
2. Future extensibility: swapping local disk for S3/GCS requires
   changes only here, not throughout the codebase.

In production, replace LocalFileStorage with an S3-backed implementation
that streams directly to/from object storage — this avoids storing files
on ephemeral container disks and enables horizontal scaling.
"""

import json
from pathlib import Path
from typing import Any, Dict

from app.config.settings import Settings
from app.utils.file_helpers import ensure_directory, safe_delete
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LocalFileStorage:
    """
    Manages the local filesystem lifecycle of upload and output files.

    Responsibilities:
    - Providing paths for upload staging and normalized audio.
    - Persisting transcription JSON results to the output directory.
    - Cleaning up temporary files after processing completes.

    This class does NOT perform I/O itself for reads (the caller writes
    bytes to the path this class provides). This keeps the class focused
    on path management and lifecycle, not I/O streaming.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize storage, ensuring required directories exist.

        Args:
            settings: Application settings with upload_dir and output_dir.
        """
        self._upload_dir = ensure_directory(Path(settings.upload_dir))
        self._output_dir = ensure_directory(Path(settings.output_dir))
        logger.info(
            "Storage initialized | upload_dir=%s output_dir=%s",
            self._upload_dir,
            self._output_dir,
        )

    # ------------------------------------------------------------------ #
    # Path provisioning
    # ------------------------------------------------------------------ #

    def get_upload_path(self, filename: str) -> Path:
        """
        Return a staging path for an incoming upload.

        The caller writes the uploaded bytes to this path. The filename
        must already be sanitized (i.e., produced by generate_unique_filename).

        Args:
            filename: Safe, unique filename (no directory components).

        Returns:
            Absolute path to the upload staging location.
        """
        return self._upload_dir / filename

    def get_normalized_path(self, filename: str) -> Path:
        """
        Return the expected path for a normalized WAV file.

        Args:
            filename: Normalized WAV filename.

        Returns:
            Path within the upload directory (normalization is ephemeral).
        """
        return self._upload_dir / filename

    def get_output_path(self, stem: str) -> Path:
        """
        Return the path for the transcription JSON output file.

        The JSON is written to the output directory for auditing and caching.
        In a production system, this would be persisted to object storage
        (S3, GCS) keyed by the file content hash.

        Args:
            stem: Base name for the output file (without extension).

        Returns:
            Path to the .json output file.
        """
        return self._output_dir / f"{stem}.json"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_result(self, stem: str, result: Dict[str, Any]) -> Path:
        """
        Persist a transcription result as a JSON file.

        Writing results to disk enables:
        1. Auditing: review past transcriptions without re-running inference.
        2. Caching: return cached results for duplicate uploads (future work).
        3. Debugging: inspect raw Whisper output independently of the API.

        Args:
            stem: Base filename for the output JSON (e.g. audio file stem).
            result: Transcription result dictionary.

        Returns:
            Path to the saved JSON file.
        """
        output_path = self.get_output_path(stem)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Transcription result saved | path=%s", output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def cleanup_upload(self, *paths: Path) -> None:
        """
        Delete temporary files created during a single request's processing.

        Called in a finally block to ensure temp files are removed even
        if transcription fails midway. Prevents unbounded disk growth in
        high-throughput deployments.

        Args:
            *paths: One or more file paths to delete.
        """
        for path in paths:
            safe_delete(path)
            logger.debug("Cleaned up temp file | path=%s", path)
