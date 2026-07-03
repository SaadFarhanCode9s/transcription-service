# ============================================================
# Speech-to-Text Transcription Service — Dockerfile
# ============================================================
#
# Multi-stage build strategy:
#   Stage 1 (builder): Install Python dependencies in a venv.
#   Stage 2 (runtime): Copy only the venv, not build tools.
#
# Benefits:
# - Smaller final image (no pip, gcc, etc.)
# - Faster layer caching: code changes don't invalidate dependency layers.
# - Clear separation between build and runtime concerns.
# ============================================================

# ---- Stage 1: Build ----------------------------------------
FROM python:3.11-slim AS builder

# Install build dependencies for packages that compile C extensions
# (numpy, tokenizers, etc.). These are NOT copied to the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first to leverage Docker layer caching.
# If requirements.txt has not changed, this layer is reused on
# subsequent builds — significantly faster for code-only changes.
COPY requirements.txt .

# Install into an isolated virtual environment so we can copy it
# cleanly to the runtime stage without polluting system Python.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: Runtime --------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="transcription-service"
LABEL description="Speech-to-Text Transcription Service powered by WhisperX/Whisper and FFmpeg"
LABEL version="1.0.0"

# Install runtime system dependencies:
# - ffmpeg: Audio normalization (required at runtime, not just build time)
# - libmagic1: MIME type detection for upload validation
# - libsndfile1: Audio file I/O for some Whisper dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libmagic1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security.
# Running as root inside a container is a significant attack surface:
# if the application is compromised, the attacker has root in the container,
# making host escape exploits more dangerous.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code.
COPY --chown=appuser:appuser . .

# Create upload and output directories with correct ownership.
RUN mkdir -p uploads output sample_audio && \
    chown -R appuser:appuser uploads output sample_audio

# Switch to non-root user before starting the server.
USER appuser

# Expose the application port.
EXPOSE 8000

# Health check: Docker Swarm and Kubernetes use this to determine
# container readiness. Runs every 30s; fails after 3 consecutive failures.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Production entrypoint.
# --workers: Match to CPU core count (1 per core is a good starting point).
# --timeout-keep-alive: Prevents idle connections from holding worker slots.
# Not using --reload in production — hot reload is for development only.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "5", \
     "--log-level", "info"]
