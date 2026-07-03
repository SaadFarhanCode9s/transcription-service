# 🛠 Developer Testing & API Verification Guide

This guide provides copy-pasteable `curl` commands to thoroughly test the Speech-to-Text Transcription Service. Each endpoint, happy path, validation guardrail, and error state is covered.

---

## 1. Prerequisites

Ensure your service is running locally before executing the tests:

```bash
# Option A: Local execution
cd transcription-service
python3 main.py

# Option B: Docker execution
cd transcription-service
docker-compose up -d
```

By default, the server runs on `http://localhost:8000`.

---

## 2. API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/` | Service description and links |
| **GET** | `/api/v1/health` | Service health status |
| **POST** | `/api/v1/transcribe` | Upload and transcribe audio file |

---

## 3. GET / (Service Root Redirect)

Verify that the root endpoint correctly redirects or returns metadata and links to the API docs.

### Command
```bash
curl -i http://localhost:8000/
```

### Expected Output Headers & Body
```http
HTTP/1.1 200 OK
content-type: application/json
content-length: 98

{
  "service": "Speech-to-Text Transcription Service",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

---

## 4. GET /api/v1/health (Health check)

Verifies the service status, loaded model parameters, and FFmpeg environment state.

### Command (Pretty JSON format)
```bash
curl -s http://localhost:8000/api/v1/health | jq .
```

### Expected Output
```json
{
  "status": "ok",
  "version": "1.0.0",
  "whisper_model": "base",
  "ffmpeg_available": true
}
```

---

## 5. POST /api/v1/transcribe (Transcription Endpoint)

### Test A: Happy Path (Valid Audio File)
Submit a valid supported audio format (WAV, MP3, FLAC, M4A, OGG, AAC) to be normalized and transcribed.

```bash
# Generate the sample WAV audio first if you haven't already
python3 scripts/generate_sample_audio.py

# Send upload request
curl -s -X POST http://localhost:8000/api/v1/transcribe \
     -F "file=@sample_audio/sample_speech.mp3" | jq .
```

#### Expected Output
```json
{
  "language": "en",
  "duration": 3.0,
  "transcription": "Hello everyone.",
  "segments": [
    {
      "start": 0.0,
      "end": 3.0,
      "text": "Hello everyone."
    }
  ]
}
```

---

### Test B: Missing File Payload (HTTP 400 / 422)
Submit a request to `/transcribe` without sending the `file` parameter in the form payload.

```bash
curl -s -w "\nHTTP Status: %{http_code}\n" \
     -X POST http://localhost:8000/api/v1/transcribe
```

#### Expected Output
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "file"],
      "msg": "Field required",
      "input": null
    }
  ]
}
HTTP Status: 422
```

---

### Test C: Unsupported File Extension (HTTP 415)
Submit a file with an unsupported format (e.g., text, document, images).

```bash
# Create dummy txt file
echo "Hello from a text file" > test_doc.txt

# Attempt to transcribe
curl -s -w "\nHTTP Status: %{http_code}\n" \
     -X POST http://localhost:8000/api/v1/transcribe \
     -F "file=@test_doc.txt" | jq .
```

#### Expected Output
```json
{
  "detail": {
    "error": "unsupported_format",
    "message": "Unsupported file format 'test_doc.txt' (MIME: text/plain). Accepted formats: wav, mp3, flac, m4a, ogg, aac.",
    "details": {
      "filename": "test_doc.txt",
      "mime_type": "text/plain",
      "allowed_extensions": ["wav", "mp3", "flac", "m4a", "ogg", "aac"]
    }
  }
}
HTTP Status: 415
```

---

### Test D: Content Spoofing Guardrail (HTTP 415)
Submit a non-audio file disguised as a WAV file (e.g., an HTML/Text file renamed to `spoof.wav`). The service checks file magic bytes instead of just looking at the extension.

```bash
# Create fake WAV file containing plain text
echo "<html><body>fake audio</body></html>" > spoof.wav

# Attempt to upload spoofed file
curl -s -w "\nHTTP Status: %{http_code}\n" \
     -X POST http://localhost:8000/api/v1/transcribe \
     -F "file=@spoof.wav" | jq .
```

#### Expected Output
```json
{
  "detail": {
    "error": "unsupported_format",
    "message": "Unsupported file format 'spoof.wav' (MIME: text/html). Accepted formats: wav, mp3, flac, m4a, ogg, aac.",
    "details": {
      "filename": "spoof.wav",
      "mime_type": "text/html",
      "allowed_extensions": ["wav", "mp3", "flac", "m4a", "ogg", "aac"]
    }
  }
}
HTTP Status: 415
```

---

### Test E: File Exceeds Size Limit (HTTP 413)
Generate a dummy test file exceeding the maximum size limit configured in the application (500 MB by default, or configured lower for validation testing) and verify that it is rejected.

For demonstration, if you configured `MAX_UPLOAD_SIZE_MB=1` in your `.env` file, try sending a 2 MB file:

```bash
# Create a dummy 2 MB file (with WAV headers to pass MIME check)
# Set size limit temporarily to 1 MB in settings to check
dd if=/dev/zero of=large_temp.wav bs=1M count=2

curl -s -w "\nHTTP Status: %{http_code}\n" \
     -X POST http://localhost:8000/api/v1/transcribe \
     -F "file=@large_temp.wav" | jq .
```

#### Expected Output
```json
{
  "detail": {
    "error": "file_too_large",
    "message": "File size 2.0 MB exceeds the maximum allowed size of 1 MB.",
    "details": {
      "size_bytes": 2097152,
      "max_bytes": 1048576
    }
  }
}
HTTP Status: 413
```

---

## 6. Advanced Testing Tips

### Clean up test resources
After finishing your manual testing, make sure to clean up the local test assets:
```bash
rm -f test_doc.txt spoof.wav large_temp.wav
```

### Inspect timing using Curl write-out format
You can print latency directly from curl using:
```bash
curl -s -w "Total Time: %{time_total}s\n" \
     -X POST http://localhost:8000/api/v1/transcribe \
     -F "file=@sample_audio/sample_speech.mp3" > /dev/null
```
This is useful for verifying cold start model loading vs. warm start cache behavior.
