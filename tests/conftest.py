"""
Shared pytest fixtures for the test suite.

Fixtures follow the principle of minimal setup: each fixture provides
exactly what a test needs, nothing more. Shared state between tests is
avoided to keep tests independent and order-agnostic.

We mock Whisper at the fixture level rather than per-test to keep
individual tests focused on what they are actually testing (HTTP behavior,
validation logic, etc.) rather than model loading details.
"""

import io
import struct
import wave
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.main import create_app
from app.services.audio_normalizer import AudioNormalizer
from app.services.audio_chunker import AudioChunker
from app.services.storage import LocalFileStorage
from app.services.transcriber import TranscriptionService
from app.services.validator import UploadValidator


# --------------------------------------------------------------------------- #
# Settings override
# --------------------------------------------------------------------------- #


def make_test_settings(tmp_path: Path) -> Settings:
    """
    Build a Settings instance with test-safe overrides.

    Uses tmp_path (pytest's per-test temp directory) for uploads and output
    so tests do not pollute the real project directories and run in parallel
    safely.
    """
    return Settings(
        upload_dir=str(tmp_path / "uploads"),
        output_dir=str(tmp_path / "output"),
        max_upload_size_mb=10,   # Small limit for tests
        whisper_model="base",
        debug=True,
        log_level="DEBUG",
    )


# --------------------------------------------------------------------------- #
# Mock transcription backend
# --------------------------------------------------------------------------- #


class MockTranscriptionBackend:
    """
    Deterministic mock backend that returns canned transcription results.

    Using a hand-written mock (rather than MagicMock) gives us explicit
    control over the return value shape, ensuring tests catch any contract
    changes between the service layer and the backend interface.
    """

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        language: str | None = None,
    ) -> tuple:
        return (
            "en",
            "Hello everyone. Welcome to the test.",
            [
                {"start": 0.0, "end": 3.0, "text": "Hello everyone."},
                {"start": 3.5, "end": 6.0, "text": "Welcome to the test."},
            ],
        )


# --------------------------------------------------------------------------- #
# Audio generation helpers
# --------------------------------------------------------------------------- #


def generate_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a minimal valid WAV file as bytes.

    Produces a sine wave at 440 Hz — audible but simple. Used by tests
    that need a real (not zero-length) audio file to avoid FFmpeg errors.

    Args:
        duration_seconds: Length of the generated audio.
        sample_rate: Sample rate in Hz.

    Returns:
        WAV file contents as bytes.
    """
    n_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    # 440 Hz sine wave, amplitude 0.3 to avoid clipping
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Pytest fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Per-test Settings with isolated temp directories."""
    return make_test_settings(tmp_path)


@pytest.fixture
def mock_backend() -> MockTranscriptionBackend:
    """Deterministic mock transcription backend."""
    return MockTranscriptionBackend()


@pytest.fixture
def transcription_service(test_settings: Settings, mock_backend: MockTranscriptionBackend) -> TranscriptionService:
    """TranscriptionService wired with the mock backend."""
    return TranscriptionService(backend=mock_backend, settings=test_settings)


@pytest.fixture
def validator(test_settings: Settings) -> UploadValidator:
    """UploadValidator using test settings."""
    return UploadValidator(test_settings)


@pytest.fixture
def storage(test_settings: Settings) -> LocalFileStorage:
    """LocalFileStorage with isolated temp directories."""
    return LocalFileStorage(test_settings)


@pytest.fixture
def chunker(test_settings: Settings) -> AudioChunker:
    """AudioChunker using test settings."""
    return AudioChunker(test_settings)


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """1-second WAV file as bytes."""
    return generate_wav_bytes(duration_seconds=1.0)


@pytest.fixture
def sample_wav_file(tmp_path: Path, sample_wav_bytes: bytes) -> Path:
    """1-second WAV file written to disk."""
    wav_path = tmp_path / "test_audio.wav"
    wav_path.write_bytes(sample_wav_bytes)
    return wav_path


@pytest.fixture
def client(tmp_path: Path, mock_backend: MockTranscriptionBackend) -> Generator[TestClient, None, None]:
    """
    TestClient with mocked transcription backend and test settings.

    We patch:
    1. get_settings() — to use temp directories.
    2. create_transcription_service() — to skip model loading.

    This allows full end-to-end HTTP tests without Whisper installed.
    """
    settings = make_test_settings(tmp_path)

    def _override_settings() -> Settings:
        return settings

    def _override_transcription_service(s: Settings) -> TranscriptionService:
        return TranscriptionService(backend=mock_backend, settings=settings)

    app = create_app()
    app.dependency_overrides[get_settings] = _override_settings

    # Patch create_transcription_service in the lifespan to skip GPU/model loading
    with patch("app.main.create_transcription_service", side_effect=_override_transcription_service):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
