"""
Application configuration management.

Using Pydantic BaseSettings here gives us:
1. Automatic environment variable parsing
2. Type validation on startup (fail fast, not at runtime)
3. A single source of truth for all configuration values
4. Easy .env file support for local development without touching code

All tunable values live here — never hardcoded in business logic.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    Values are validated at application startup so misconfiguration fails
    immediately rather than at the moment a feature is exercised.
    """

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_name: str = Field(default="Speech-to-Text Transcription Service", description="Human-readable service name")
    app_version: str = Field(default="1.0.0", description="Semantic version")
    debug: bool = Field(default=False, description="Enable debug mode (never True in production)")
    log_level: str = Field(default="INFO", description="Python logging level")
    environment: str = Field(default="development", description="Runtime environment: development | staging | production")

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, description="Bind port")
    workers: int = Field(default=1, description="Uvicorn worker count; increase with CPU cores in production")

    # ------------------------------------------------------------------ #
    # File handling
    # ------------------------------------------------------------------ #
    upload_dir: Path = Field(default=Path("uploads"), description="Temporary directory for raw uploads")
    output_dir: Path = Field(default=Path("output"), description="Directory for processed audio and transcription JSON")

    # Maximum upload size: 500 MB is generous for long audio files.
    # Production systems should additionally enforce this at the reverse proxy
    # (e.g. nginx client_max_body_size) to reject oversized payloads before
    # they reach Python.
    max_upload_size_mb: int = Field(default=500, description="Maximum upload size in megabytes")

    # Supported MIME types mirror the extensions list below.
    # Both extension AND MIME type are validated to prevent extension spoofing.
    allowed_extensions: List[str] = Field(
        default=["wav", "mp3", "flac", "m4a", "ogg", "aac"],
        description="Accepted audio file extensions (lowercase, no dot)",
    )
    allowed_mime_types: List[str] = Field(
        default=[
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
            "audio/mp3",
            "audio/flac",
            "audio/x-flac",
            "audio/mp4",
            "audio/x-m4a",
            "audio/ogg",
            "audio/aac",
            "audio/x-aac",
            "video/mp4",  # M4A files are sometimes reported as video/mp4
        ],
        description="Accepted MIME types for uploaded audio",
    )

    # ------------------------------------------------------------------ #
    # FFmpeg audio normalization
    # ------------------------------------------------------------------ #
    # 16 kHz mono is the Whisper training format. Feeding any other format
    # causes Whisper to resample internally, introducing inconsistency.
    # We normalize upfront so the transcription step always receives the
    # exact format the model was designed for.
    audio_sample_rate: int = Field(default=16000, description="Target sample rate in Hz (Whisper is trained at 16 kHz)")
    audio_channels: int = Field(default=1, description="Target channel count (mono)")
    audio_bit_depth: str = Field(default="pcm_s16le", description="PCM encoding for WAV output")

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    # Long audio files (>30 s) must be chunked for two reasons:
    # 1. Whisper's context window is 30 seconds of audio. Audio beyond that
    #    boundary is truncated or poorly transcribed if fed as one blob.
    # 2. Memory efficiency: loading a 2-hour WAV into numpy at once can
    #    exhaust RAM on modest hardware.
    # Overlap prevents words near chunk boundaries from being cut mid-utterance.
    chunk_duration_seconds: int = Field(default=30, description="Audio chunk length in seconds")
    chunk_overlap_seconds: int = Field(default=2, description="Overlap between consecutive chunks to avoid boundary word loss")

    # ------------------------------------------------------------------ #
    # Whisper / WhisperX
    # ------------------------------------------------------------------ #
    # "base" balances speed vs. accuracy. For production transcription
    # requiring high accuracy, use "small", "medium", or "large-v3".
    # Larger models require more VRAM/RAM and run slower on CPU.
    whisper_model: str = Field(default="base", description="Whisper model size: tiny | base | small | medium | large-v3")
    whisper_device: str = Field(default="cpu", description="Inference device: cpu | cuda | mps")
    whisper_compute_type: str = Field(default="int8", description="Quantization type for WhisperX: int8 | float16 | float32")
    whisper_language: str = Field(default="", description="Force language code (e.g. 'en'). Empty = auto-detect")
    whisper_task: str = Field(default="transcribe", description="Whisper task: transcribe | translate")

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #
    @property
    def max_upload_size_bytes(self) -> int:
        """Bytes equivalent of max_upload_size_mb, used for byte-level comparisons."""
        return self.max_upload_size_mb * 1024 * 1024

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Reject invalid log levels at startup."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return upper

    @field_validator("whisper_task")
    @classmethod
    def validate_whisper_task(cls, v: str) -> str:
        """Whisper only supports two tasks; fail early if misconfigured."""
        if v not in {"transcribe", "translate"}:
            raise ValueError(f"whisper_task must be 'transcribe' or 'translate', got '{v}'")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Allow extra fields so future .env additions don't break old code
        "extra": "ignore",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    lru_cache ensures we parse the environment exactly once per process
    lifetime. This avoids repeated disk I/O for every request and is safe
    because environment variables do not change during runtime.
    """
    return Settings()
