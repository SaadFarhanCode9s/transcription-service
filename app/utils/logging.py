"""
Structured logging configuration.

We use Python's standard `logging` module rather than a third-party library
to keep dependencies minimal. The formatter outputs JSON-style key=value pairs
so log aggregators (ELK, Datadog, CloudWatch) can parse fields without regex.

In production, swap the StreamHandler for a handler that ships logs to your
centralized logging infrastructure (e.g. fluent-bit, loguru with a sink, etc.).
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator


def configure_logging(log_level: str = "INFO") -> None:
    """
    Set up root logger with a structured formatter.

    Called once at application startup. All subsequent `logging.getLogger`
    calls inherit this configuration.

    Args:
        log_level: Python logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Use a custom formatter that includes timestamp, level, logger name,
    # and the message in a structured format easy to grep and parse.
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Remove any pre-existing handlers to avoid duplicate log entries
    # when this function is called multiple times (e.g., during tests).
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress overly verbose third-party loggers that flood the output
    # at INFO level — we only want to see their warnings and errors.
    for noisy_logger in ("uvicorn.access", "httpx", "httpcore", "multipart"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing file | path=%s size_bytes=%d", path, size)

    Args:
        name: Logger name, conventionally `__name__` of the calling module.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str) -> Generator[None, None, None]:
    """
    Context manager that logs how long a block of code takes to execute.

    Timing information is critical for detecting performance regressions
    in transcription pipelines without requiring a full profiler.

    Usage:
        with log_execution_time(logger, "ffmpeg_normalization"):
            normalize_audio(...)

    Args:
        logger: Logger instance to write timing to.
        operation: Human-readable name of the operation being timed.

    Yields:
        None — purely a timing side-effect.
    """
    start = time.perf_counter()
    logger.info("Starting operation | op=%s", operation)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Completed operation | op=%s elapsed_seconds=%.3f", operation, elapsed)
