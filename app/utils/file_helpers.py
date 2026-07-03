"""
File handling utilities: safe filename generation and path management.

These helpers are intentionally side-effect-free (pure functions where
possible) so they are trivially testable and composable.
"""

import hashlib
import time
import uuid
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)


def generate_unique_filename(original_name: str, suffix: str = "") -> str:
    """
    Generate a collision-resistant filename that retains the original extension.

    We combine a UUID4 (random) with a timestamp to ensure uniqueness even
    when multiple uploads arrive within the same millisecond. The original
    filename is not preserved in the path — only its extension — to avoid
    directory traversal vulnerabilities from attacker-controlled filenames
    such as '../../etc/passwd.wav'.

    Args:
        original_name: The original uploaded filename (used only to extract extension).
        suffix: Optional suffix appended before the extension (e.g. "_normalized").

    Returns:
        A safe, unique filename string.
    """
    extension = Path(original_name).suffix.lower()
    unique_id = uuid.uuid4().hex[:12]  # 12 hex chars = 48 bits of entropy; collision-free in practice
    timestamp = int(time.time() * 1000)  # Millisecond precision for ordering
    return f"{timestamp}_{unique_id}{suffix}{extension}"


def ensure_directory(path: Path) -> Path:
    """
    Create a directory (and parents) if it does not exist.

    Using parents=True and exist_ok=True is idempotent — safe to call
    on every request startup without checking existence first.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The same path (allows method chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute a hex-digest hash of a file's contents.

    Used for content-addressable deduplication: if two uploads produce the
    same hash, we can skip re-processing and return a cached result.
    Reading in 64 KB chunks avoids loading the entire file into RAM.

    Args:
        file_path: Path to the file to hash.
        algorithm: Hash algorithm name (sha256, md5, etc.).

    Returns:
        Lowercase hex-digest string.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_delete(path: Path) -> None:
    """
    Delete a file, silently ignoring FileNotFoundError.

    Cleanup code should never crash the request if a temp file was already
    deleted by another process or a previous cleanup attempt.

    Args:
        path: File path to delete.
    """
    try:
        path.unlink(missing_ok=True)
        logger.debug("Deleted temporary file | path=%s", path)
    except OSError as exc:
        # Log but do not re-raise — cleanup failure is non-fatal.
        logger.warning("Could not delete file | path=%s error=%s", path, exc)
