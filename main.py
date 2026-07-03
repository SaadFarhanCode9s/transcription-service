"""
Application entrypoint — runs the FastAPI server via uvicorn.

Usage:
    python main.py                    # Development (single worker, auto-reload)
    ENVIRONMENT=production python main.py   # Production (no reload)

In production, prefer running uvicorn directly via Docker CMD for better
signal handling and process management:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import uvicorn

from app.config.settings import get_settings
from app.utils.logging import configure_logging


def main() -> None:
    """Configure and start the uvicorn server."""
    settings = get_settings()
    configure_logging(settings.log_level)

    is_dev = settings.environment != "production"

    uvicorn.run(
        # Use the import string (not the app object) when reload=True.
        # uvicorn needs to re-import the module on file changes, which
        # requires the string form rather than a direct reference.
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1 if is_dev else settings.workers,
        reload=is_dev,   # Hot reload in development only — incompatible with multiple workers
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
