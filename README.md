# 🎙 Speech-to-Text Transcription Service

> A production-ready REST API for audio transcription powered by **WhisperX** / **OpenAI Whisper** and **FFmpeg**.  
> Accepts WAV, MP3, FLAC, M4A, OGG, and AAC. Returns timestamped segments with language detection.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Quick Start](#quick-start)
  - [Local Installation](#local-installation)
  - [Docker](#docker)
- [API Documentation](#api-documentation)
- [Example Responses](#example-responses)
- [Error Responses](#error-responses)
- [Configuration Reference](#configuration-reference)
- [Design Decisions](#design-decisions)
- [Scalability](#scalability)
- [Security](#security)
- [Performance](#performance)
- [Testing](#testing)
- [Future Improvements](#future-improvements)

---

## Project Overview

This service provides a clean, well-structured REST API for converting speech audio into text. It is designed to be **deployable, testable, and maintainable** — not just functional.

Key design goals:
- **Separation of concerns**: each layer (API, validation, normalization, transcription, storage) is independently testable.
- **Fail fast**: misconfigurations and invalid inputs are caught as early as possible with clear error messages.
- **Graceful degradation**: WhisperX is used when available; the service automatically falls back to OpenAI Whisper without configuration changes.
- **Production-ready defaults**: structured logging, health checks, Docker multi-stage builds, non-root containers, and resource limits are all included out of the box.

---

## Features

| Feature | Details |
|---|---|
| **Supported formats** | WAV, MP3, FLAC, M4A, OGG, AAC |
| **Upload validation** | Presence · Extension · MIME type (magic bytes) · Size limit |
| **Audio normalization** | FFmpeg → 16 kHz · Mono · PCM WAV |
| **Transcription** | WhisperX (preferred) with OpenAI Whisper fallback |
| **Long audio support** | Configurable chunking with overlap to prevent boundary word loss |
| **Timestamped output** | Segment-level timestamps; word-level with WhisperX |
| **Language detection** | Automatic ISO 639-1 detection (or forced via config) |
| **Error handling** | Typed exceptions mapped to correct HTTP status codes |
| **Structured logging** | Key=value format with execution timing |
| **OpenAPI docs** | Auto-generated at `/docs` and `/redoc` |
| **Health check** | `/api/v1/health` for Kubernetes probes |
| **Docker** | Multi-stage build, non-root user, model cache persistence |

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **API framework** | FastAPI | Async-native, automatic OpenAPI, Pydantic integration, high I/O throughput |
| **ASGI server** | Uvicorn | Production-grade ASGI server with multi-worker support |
| **Data validation** | Pydantic v2 | Type-safe models, environment parsing, OpenAPI schema generation |
| **Audio processing** | FFmpeg | Universal codec support, battle-tested, single binary for all formats |
| **Transcription** | WhisperX / OpenAI Whisper | State-of-the-art open-source ASR, no API cost, offline operation |
| **MIME detection** | python-magic (libmagic) | Content-aware type detection prevents extension-spoofing |
| **Testing** | pytest + pytest-cov | Parametric tests, fixtures, coverage enforcement |
| **Containerization** | Docker + Docker Compose | Reproducible environments, dependency isolation |

---

## Architecture

### Request Flow

```
┌─────────────────────────────────────────────────────────┐
│                        CLIENT                           │
│              curl / Postman / Web App                   │
└──────────────────────────┬──────────────────────────────┘
                           │  POST /api/v1/transcribe
                           │  multipart/form-data
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI / Uvicorn                     │
│              (Routing + Middleware + DI)                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Upload Staging                         │
│        Stream upload to disk in 64 KB chunks            │
│        (avoids loading entire file into RAM)            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  UploadValidator                         │
│  ① Filename present?                                    │
│  ② Extension in allow-list?                             │
│  ③ MIME type via libmagic (magic byte inspection)?      │
│  ④ File size ≤ configured maximum?                      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  AudioNormalizer (FFmpeg)                │
│  Convert to: 16 kHz · Mono · PCM WAV                   │
│  Drops video streams (e.g. M4A cover art)              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  AudioChunker                           │
│  Load WAV into numpy float32 array                      │
│  Split into 30s chunks with 2s overlap                  │
│  (single chunk if audio < 30s)                          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              TranscriptionService (per chunk)           │
│                                                         │
│  ┌─────────────┐         ┌──────────────────────────┐  │
│  │  WhisperX   │  ─OR─   │  OpenAI Whisper          │  │
│  │  (preferred)│         │  (fallback)              │  │
│  │  INT8 quant │         │  FP32 CPU inference      │  │
│  │  word-level │         │  segment-level timestamps│  │
│  │  timestamps │         │                          │  │
│  └─────────────┘         └──────────────────────────┘  │
│                                                         │
│  → Adjust timestamps by chunk offset                    │
│  → Remove overlap tail (prevent duplicates)             │
│  → Merge all chunks into single result                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               LocalFileStorage                          │
│   Persist transcription JSON to /output/                │
│   (future: swap for S3/GCS without changing API)        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                JSON Response                            │
│  { language, duration, transcription, segments }        │
└─────────────────────────────────────────────────────────┘
                           │
                    ALWAYS (finally block)
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Cleanup                               │
│   Delete raw upload + normalized WAV from /uploads/     │
└─────────────────────────────────────────────────────────┘
```

### Exception → HTTP Status Code Mapping

| Exception | HTTP Code | Meaning |
|---|---|---|
| `FileMissingError` | 400 Bad Request | No file in the request |
| `UnsupportedFormatError` | 415 Unsupported Media Type | Wrong extension or MIME type |
| `FileTooLargeError` | 413 Request Entity Too Large | File exceeds size limit |
| `AudioNormalizationError` | 422 Unprocessable Entity | File valid but FFmpeg can't decode it |
| `TranscriptionError` | 500 Internal Server Error | Model inference failed |

---

## Folder Structure

```
transcription-service/
│
├── app/                        # Application source code
│   ├── api/
│   │   └── routes.py           # FastAPI route definitions (thin — no business logic)
│   ├── config/
│   │   └── settings.py         # Pydantic BaseSettings — all configuration lives here
│   ├── models/
│   │   └── response_models.py  # Pydantic models for API request/response contracts
│   ├── services/
│   │   ├── audio_chunker.py    # Splits long audio into overlapping chunks
│   │   ├── audio_normalizer.py # FFmpeg wrapper: converts to 16kHz mono PCM WAV
│   │   ├── storage.py          # File lifecycle management (upload → output → cleanup)
│   │   ├── transcriber.py      # Whisper orchestration: chunk → transcribe → merge
│   │   └── validator.py        # Upload validation: presence, extension, MIME, size
│   ├── utils/
│   │   ├── exceptions.py       # Typed exception hierarchy with HTTP code semantics
│   │   ├── file_helpers.py     # Safe filename generation, directory utilities
│   │   └── logging.py          # Structured logging config + timing context manager
│   └── main.py                 # FastAPI app factory + lifespan (startup/shutdown)
│
├── tests/
│   ├── conftest.py             # Shared fixtures: mock backend, WAV generator, TestClient
│   ├── test_api.py             # HTTP endpoint integration tests
│   ├── test_chunker.py         # AudioChunker unit tests
│   ├── test_transcription_service.py  # TranscriptionService unit tests
│   └── test_validator.py       # UploadValidator unit tests
│
├── scripts/
│   └── generate_sample_audio.py  # Generates a test WAV file for manual testing
│
├── sample_audio/
│   └── sample_output.json      # Example API response for reference
│
├── uploads/                    # Temporary staging area (auto-cleaned per request)
├── output/                     # Persisted transcription JSON files
│
├── main.py                     # Entrypoint: python main.py
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Local/single-host deployment
├── requirements.txt            # Python dependencies (with rationale comments)
├── pyproject.toml              # pytest + coverage configuration
├── .env.example                # Documented environment variable template
├── .gitignore                  # Excludes secrets, models, runtime artifacts
└── README.md                   # This file
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg installed on PATH
- `libmagic` installed (for MIME type detection)

#### Install FFmpeg

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg libmagic1

# macOS (Homebrew)
brew install ffmpeg libmagic

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

---

### Local Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd transcription-service

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install a transcription backend (choose one or both)

# Option A — WhisperX (recommended: faster, word-level timestamps)
pip install whisperx

# Option B — OpenAI Whisper (fallback, simpler installation)
pip install openai-whisper

# 5. Configure the service
cp .env.example .env
# Edit .env to set WHISPER_MODEL, WHISPER_DEVICE, etc.

# 6. Run the service
python main.py
```

The API is now available at **http://localhost:8000**.

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/api/v1/health

---

### Docker

```bash
# Build the image
docker-compose build

# Start the service in the foreground (see logs)
docker-compose up

# Start in the background
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (clears model cache and output)
docker-compose down -v
```

> **First startup note**: Whisper downloads the model on first run (~74 MB for `base`). This is cached in the `whisper_cache` Docker volume — subsequent starts use the cache.

---

## API Documentation

### POST /api/v1/transcribe

Transcribe an audio file.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | ✅ | Audio file (WAV, MP3, FLAC, M4A, OGG, AAC) |

**curl**
```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
     -F "file=@path/to/audio.mp3"
```

**curl with verbose output**
```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
     -F "file=@audio.wav" \
     -H "Accept: application/json" \
     -w "\nHTTP Status: %{http_code}\n"
```

**Python (requests)**
```python
import requests

with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/transcribe",
        files={"file": ("audio.mp3", f, "audio/mpeg")},
    )

result = response.json()
print(f"Language: {result['language']}")
print(f"Duration: {result['duration']}s")
print(f"Text: {result['transcription']}")
```

**Postman**
1. Set method to **POST**, URL to `http://localhost:8000/api/v1/transcribe`
2. Select **Body** → **form-data**
3. Add key `file`, change type from **Text** to **File**
4. Select your audio file
5. Click **Send**

---

### GET /api/v1/health

Service health and readiness check.

```bash
curl http://localhost:8000/api/v1/health
```

Used by:
- Kubernetes liveness probes (`failureThreshold: 3`)
- Kubernetes readiness probes (`initialDelaySeconds: 60`)
- Load balancers
- Uptime monitoring (Datadog, Pingdom)

---

## Example Responses

### Successful Transcription (HTTP 200)

```json
{
  "language": "en",
  "duration": 31.5,
  "transcription": "Hello everyone. Welcome to this demonstration of the transcription service.",
  "segments": [
    {
      "start": 0.00,
      "end": 3.25,
      "text": "Hello everyone."
    },
    {
      "start": 3.50,
      "end": 8.10,
      "text": "Welcome to this demonstration of the transcription service."
    }
  ]
}
```

### Health Check (HTTP 200)

```json
{
  "status": "ok",
  "version": "1.0.0",
  "whisper_model": "base",
  "ffmpeg_available": true
}
```

### Health Check — Degraded (HTTP 503)

```json
{
  "status": "degraded",
  "version": "1.0.0",
  "whisper_model": "base",
  "ffmpeg_available": false
}
```

---

## Error Responses

All errors follow a consistent JSON envelope:

```json
{
  "detail": {
    "error": "machine_readable_code",
    "message": "Human-readable description of what went wrong.",
    "details": {}
  }
}
```

### 400 — File Missing

```bash
curl -X POST http://localhost:8000/api/v1/transcribe
```
```json
{
  "detail": {
    "error": "file_missing",
    "message": "No audio file was included in the request. Send the file under the 'file' form-data field.",
    "details": {}
  }
}
```

### 413 — File Too Large

```json
{
  "detail": {
    "error": "file_too_large",
    "message": "File size 520.0 MB exceeds the maximum allowed size of 500 MB.",
    "details": {
      "size_bytes": 545259520,
      "max_bytes": 524288000
    }
  }
}
```

### 415 — Unsupported Format

```json
{
  "detail": {
    "error": "unsupported_format",
    "message": "Unsupported file format 'audio.xyz' (MIME: application/octet-stream). Accepted formats: wav, mp3, flac, m4a, ogg, aac.",
    "details": {
      "filename": "audio.xyz",
      "mime_type": "application/octet-stream",
      "allowed_extensions": ["wav", "mp3", "flac", "m4a", "ogg", "aac"]
    }
  }
}
```

### 422 — Unprocessable Audio

```json
{
  "detail": {
    "error": "audio_normalization_error",
    "message": "Failed to normalize audio file 'corrupt.wav': FFmpeg exited with code 1: Invalid data found when processing input",
    "details": {
      "filename": "corrupt.wav",
      "reason": "FFmpeg exited with code 1: Invalid data found when processing input"
    }
  }
}
```

---

## Configuration Reference

All configuration is managed via environment variables (or `.env` file). See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Runtime environment |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `WORKERS` | `1` | Uvicorn worker processes |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload size |
| `WHISPER_MODEL` | `base` | Model size (tiny/base/small/medium/large-v3) |
| `WHISPER_DEVICE` | `cpu` | Inference device (cpu/cuda) |
| `WHISPER_COMPUTE_TYPE` | `int8` | WhisperX quantization type |
| `WHISPER_LANGUAGE` | _(empty)_ | Force language; empty = auto-detect |
| `CHUNK_DURATION_SECONDS` | `30` | Audio chunk size for long files |
| `CHUNK_OVERLAP_SECONDS` | `2` | Overlap between adjacent chunks |

---

## Design Decisions

### Why FastAPI?

FastAPI's async-native design means file upload I/O does not block the event loop. Its automatic OpenAPI generation eliminates separate documentation maintenance — the docs are always in sync with the code because they are derived from type hints. Pydantic v2 integration provides request validation with essentially zero boilerplate compared to Flask/Django.

Compared to Flask: FastAPI's dependency injection system makes testing dramatically easier — no application context, no global state, just inject a mock.

Compared to Django: Django is optimized for traditional web applications with database-driven views. A transcription microservice has no models in the Django sense, making Django's overhead unjustified.

### Why WhisperX with OpenAI Whisper fallback?

**WhisperX** uses CTranslate2's quantized inference, running 3–4× faster than the original model on CPU while maintaining comparable accuracy. It also provides word-level timestamps via forced phoneme alignment — significantly more accurate than Whisper's default segment-level timing.

**OpenAI Whisper** is the fallback because it has a simpler installation (just `pip install openai-whisper`) and works on all platforms without native library dependencies beyond PyTorch.

The service uses Python's duck typing (a `Protocol` interface) so the caller never needs to know which backend is active. This is the [Strategy pattern](https://refactoring.guru/design-patterns/strategy) in practice.

### Why FFmpeg?

FFmpeg handles every audio format and codec combination, including edge cases like multi-stream MP4, VBR MP3, and M4A with cover art embedded as a video stream. The alternative — using per-format Python libraries (`pydub`, `librosa`, `soundfile`) — creates a fragile dependency matrix and produces inconsistent behavior across formats.

Using a subprocess call rather than a Python binding gives us complete access to all FFmpeg flags and makes the command reproducible and debuggable outside the application.

### Why 16 kHz mono PCM?

Whisper was trained exclusively on 16 kHz mono audio. The model internally resamples any other input, but resampling twice (once by us, once by Whisper) introduces small artifacts. Normalizing once, to exactly the training format, eliminates this inconsistency.

PCM (uncompressed) allows numpy to memory-map the file directly without a decoding step, reducing CPU overhead during the already-expensive inference phase.

### Why chunking?

1. **Context window**: Whisper's attention mechanism has a fixed 30-second receptive field. Audio beyond this boundary gets less context, degrading accuracy.
2. **Memory**: A 2-hour WAV at 16 kHz mono requires ~230 MB as a numpy float32 array. Multiple concurrent requests would exhaust RAM on modest hardware.
3. **Parallelization**: Independent chunks can be processed concurrently on multiple CPU cores or GPU workers (future optimization).
4. **Fault isolation**: A failed chunk can be retried without reprocessing the entire file.

### Why a typed exception hierarchy?

String-based error detection (`if "404" in error_message`) is brittle. A dedicated exception for each error condition (`FileMissingError`, `FileTooLargeError`, etc.) means:
- The API layer catches by type and maps to exact HTTP status codes without parsing messages.
- Each exception carries a machine-readable `code` field for programmatic client handling.
- New error types can be added without touching existing handlers.

### Why modular architecture?

Each layer (`validator`, `audio_normalizer`, `audio_chunker`, `transcriber`, `storage`) has a single responsibility and depends only on its configuration and lower-level abstractions. This means:
- Each module is testable in isolation (no need to start a server to test chunking logic).
- Replacing a component (e.g., swapping local disk storage for S3) requires changes to exactly one class.
- New developers can read and understand one layer at a time.

### Why persist output JSON?

Transcription is expensive (10–60× real-time on CPU). Persisting results enables:
1. **Auditing**: Review past transcriptions for quality monitoring.
2. **Caching**: Return cached results for duplicate uploads (content-hash lookup — future work).
3. **Debugging**: Inspect raw Whisper output independently of the API layer.

---

## Scalability

The current single-process architecture is appropriate for:
- Development and testing
- Low-volume production (<100 requests/day on modest hardware)

### Path to Production Scale

```
Current (MVP)
─────────────
Client → FastAPI (single worker) → Whisper (CPU)

Phase 1: Async queue
────────────────────
Client → FastAPI → Redis (task queue) → Celery worker(s) → Whisper
                     └── Return job ID immediately
                     └── Poll /jobs/{id} for result

Phase 2: Horizontal scaling
───────────────────────────
Clients → Load Balancer (nginx / AWS ALB)
              ↓
    ┌─────────┴─────────┐
    API Pod 1      API Pod 2     (stateless FastAPI; add pods freely)
    ↓                   ↓
    Redis (shared queue + result cache)
    ↓
    ┌─────────┴─────────┐
Worker Pod 1      Worker Pod 2   (GPU workers; scale based on queue depth)
    ↓                   ↓
    S3 / GCS (shared object storage for uploads and results)

Phase 3: GPU acceleration
─────────────────────────
- Use NVIDIA T4/A10G instances with CUDA
- WhisperX with float16 compute type: 10–50× faster than CPU
- Autoscale worker pods based on GPU utilization

Phase 4: Global distribution
─────────────────────────────
- Deploy to multiple regions (AWS us-east-1, eu-west-1, ap-southeast-1)
- Route requests to the nearest region for lowest latency
- Use Kubernetes HPA (Horizontal Pod Autoscaler) based on queue depth
```

**Key infrastructure additions**:

| Component | Purpose |
|---|---|
| **Redis** | Task queue backend for Celery; result caching |
| **Celery** | Distributed task execution; GPU worker management |
| **RabbitMQ** | Alternative to Redis for task queuing (better message guarantees) |
| **Kubernetes** | Container orchestration; auto-scaling; rolling deployments |
| **S3 / GCS** | Shared object storage for uploads and results (stateless workers) |
| **GPU workers** | CUDA inference for 10–50× speed improvement |
| **Prometheus + Grafana** | Metrics collection and visualization |
| **OpenTelemetry** | Distributed tracing across services |

---

## Security

### Input Validation
- **Extension allow-list**: Explicit format whitelist; unknown extensions are rejected before reading file content.
- **MIME type inspection**: python-magic reads file magic bytes (not headers) to prevent extension spoofing attacks.
- **Size limits**: Enforced at application layer; recommend also setting `client_max_body_size` in nginx to reject large payloads before they reach Python.

### File Handling
- **Safe filenames**: Uploaded filenames are never used directly on the filesystem. A UUID + timestamp is generated to prevent path traversal attacks (`../../etc/passwd.wav`).
- **Temp file cleanup**: Upload and normalized files are always deleted after processing, even on error, preventing disk exhaustion.

### Container Security
- **Non-root user**: The container runs as `appuser`, not `root`, reducing the blast radius of a container escape.
- **Read-only source**: Only `uploads/` and `output/` directories are writable; source code is read-only.

### Future Security Additions
- **Authentication**: API key header or JWT Bearer token (FastAPI has excellent OAuth2/JWT support).
- **Rate limiting**: `slowapi` (FastAPI middleware) or nginx `limit_req_zone`.
- **Virus scanning**: ClamAV scan on upload before processing (critical for publicly-accessible APIs).
- **Content Security Policy**: For any web UI.

---

## Performance

### Chunking
Audio is split into 30s chunks, each processed independently. For files with non-speech regions (silence, music), chunks can be short-circuited early — Whisper's voice activity detection will produce empty segments quickly.

### Model Loading
The Whisper model is loaded **once at startup** and reused across all requests. Model loading takes 1–10 seconds depending on size; per-request loading would be catastrophic for throughput.

### Caching (Future)
Files are hashed (SHA-256) on upload. If a hash matches a previous result in the output directory, the cached JSON is returned immediately without re-running inference. This reduces load by 100% for duplicate uploads.

### Streaming Upload
Uploads are read in 64 KB chunks rather than loading the entire file into memory before writing to disk. This bounds memory usage during upload at O(chunk_size), not O(file_size).

### Parallel Chunk Processing (Future)
Independent audio chunks can be processed concurrently using `concurrent.futures.ThreadPoolExecutor` (CPU-bound) or distributed across Celery workers. The current serial implementation is simpler and sufficient for low-concurrency deployments.

---

## Testing

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_validator.py -v

# Run with coverage report
pytest --cov=app --cov-report=html

# Run only fast tests (exclude integration tests that need FFmpeg/Whisper)
pytest -m "not integration" -v
```

### Test Structure

| Test File | What It Tests |
|---|---|
| `test_validator.py` | Presence, extension, MIME type, size validation rules |
| `test_chunker.py` | Chunk count, timestamps, overlap coverage, sample coverage |
| `test_transcription_service.py` | Timestamp adjustment, deduplication, end-to-end output contract |
| `test_api.py` | HTTP endpoints: success, missing file, wrong format, size limit, OpenAPI schema |

All transcription backend calls are mocked — tests run without Whisper or FFmpeg installed.

---

## Future Improvements

| Feature | Effort | Value |
|---|---|---|
| Speaker diarization | Medium | High — attribute transcript segments to individual speakers |
| Streaming transcription | High | High — return partial results as audio is uploaded |
| WebSocket interface | Medium | Medium — real-time transcription of live audio streams |
| GPU inference support | Low | High — 10–50× speed improvement with CUDA |
| Async task queue | Medium | High — non-blocking requests; return job ID immediately |
| Result caching | Low | High — skip re-transcription of duplicate uploads |
| Authentication | Low | High — API key or JWT for access control |
| Prometheus metrics | Low | Medium — request count, latency, error rate, queue depth |
| OpenTelemetry tracing | Medium | Medium — distributed trace across API → worker → model |
| Cloud deployment | Medium | High — AWS ECS, GKE, or Azure AKS with GPU node pools |
| Word-level timestamps | Done (WhisperX) | — |
| Confidence scores | Medium | Low — per-segment confidence to flag low-quality transcriptions |
| Custom vocabulary | High | Medium — bias model toward domain-specific terms |

---

## License

MIT License — see `LICENSE` for details.
