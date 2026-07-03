"""
Whisper transcription backend with WhisperX → OpenAI Whisper fallback.

BACKEND SELECTION STRATEGY
---------------------------
We attempt to import WhisperX first because it offers:
  - Word-level timestamps (vs. segment-level in vanilla Whisper)
  - Faster inference via CTranslate2 quantization
  - Better alignment via forced phoneme alignment

If WhisperX is not installed (e.g., in a lightweight deployment), we fall
back to the original openai-whisper library transparently. The caller
does not need to know which backend is active — the output schema is
identical.

This approach follows the "batteries included" principle: the service
works out of the box with the simpler backend and upgrades silently when
the more capable backend is available.

THREADING / MODEL LOADING
--------------------------
Loading a Whisper model from disk takes 1–10 seconds depending on model
size. We load once at startup (via get_transcription_service()) and reuse
the loaded model across all requests. This is safe because numpy/PyTorch
inference is stateless (no mutable model state per request).

In a multi-worker uvicorn setup, each worker process loads its own model
copy. For GPU deployments, use a single worker per GPU and scale
horizontally rather than vertically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from app.config.settings import Settings
from app.models.response_models import SegmentModel
from app.services.audio_chunker import AudioChunk, AudioChunker
from app.utils.exceptions import ModelLoadError, TranscriptionError
from app.utils.logging import get_logger, log_execution_time

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Backend abstraction
# --------------------------------------------------------------------------- #


class WhisperBackend(Protocol):
    """
    Structural protocol defining the interface every backend must satisfy.

    Using Protocol (duck typing) rather than an abstract base class means
    we can adapt third-party libraries without inheritance gymnastics.
    """

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        language: str | None = None,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Transcribe a single audio chunk.

        Args:
            audio: Float32 numpy array at 16 kHz mono.
            language: Optional BCP-47 language code to force. None = auto.

        Returns:
            Tuple of (detected_language, full_text, segments).
            Each segment is a dict with keys: start, end, text.
        """
        ...


# --------------------------------------------------------------------------- #
# WhisperX backend
# --------------------------------------------------------------------------- #


class WhisperXBackend:
    """
    Transcription backend using WhisperX (faster-whisper + CTranslate2).

    WhisperX uses quantized (INT8/FP16) CTranslate2 inference, which runs
    3–4× faster than vanilla Whisper on CPU and produces word-level
    timestamps via forced alignment — a significant accuracy improvement
    for timestamp-sensitive applications.
    """

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        """
        Load the WhisperX model.

        Args:
            model_name: Whisper model identifier (tiny, base, small, etc.).
            device: Inference device (cpu, cuda).
            compute_type: Quantization type (int8, float16, float32).

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        try:
            import whisperx  # type: ignore[import]

            logger.info("Loading WhisperX model | model=%s device=%s compute=%s", model_name, device, compute_type)
            self._model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
            self._whisperx = whisperx
            logger.info("WhisperX model loaded successfully")
        except ImportError as exc:
            raise ModelLoadError(model_name, f"WhisperX not installed: {exc}") from exc
        except Exception as exc:
            raise ModelLoadError(model_name, str(exc)) from exc

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        language: str | None = None,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Transcribe one audio chunk using WhisperX.

        WhisperX returns segments with start/end timestamps already aligned
        to word boundaries via forced phoneme alignment, so no post-processing
        is required to correct timestamp drift.

        Args:
            audio: Float32 PCM array at 16 kHz mono.
            language: Force language or None for auto-detection.

        Returns:
            (detected_language, full_text, segments)
        """
        options: Dict[str, Any] = {}
        if language:
            options["language"] = language

        result = self._model.transcribe(audio, batch_size=16, **options)

        detected_language = result.get("language", "unknown")
        segments_raw = result.get("segments", [])

        # Optionally run forced alignment for word-level timestamps.
        # This adds latency but produces much more accurate per-word timing.
        try:
            align_model, metadata = self._whisperx.load_align_model(
                language_code=detected_language,
                device="cpu",
            )
            aligned = self._whisperx.align(
                segments_raw,
                align_model,
                metadata,
                audio,
                "cpu",
                return_char_alignments=False,
            )
            segments_raw = aligned.get("segments", segments_raw)
        except Exception as align_exc:
            # Alignment is a best-effort enhancement; fall back to unaligned segments
            # gracefully rather than failing the entire transcription.
            logger.warning("Forced alignment failed, using unaligned segments | error=%s", align_exc)

        segments = self._normalize_segments(segments_raw)
        full_text = " ".join(s["text"].strip() for s in segments if s.get("text"))
        return detected_language, full_text, segments

    @staticmethod
    def _normalize_segments(raw: List[Dict]) -> List[Dict[str, Any]]:
        """Ensure every segment has the expected keys with correct types."""
        normalized = []
        for seg in raw:
            normalized.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": str(seg.get("text", "")).strip(),
                }
            )
        return normalized


# --------------------------------------------------------------------------- #
# OpenAI Whisper backend (fallback)
# --------------------------------------------------------------------------- #


class OpenAIWhisperBackend:
    """
    Transcription backend using the original openai-whisper library.

    This is the fallback when WhisperX is unavailable. It produces segment-
    level (not word-level) timestamps and runs slower than WhisperX due to
    FP32 computation, but requires only a single pip install with no native
    library dependencies beyond torch.
    """

    def __init__(self, model_name: str, device: str) -> None:
        """
        Load the OpenAI Whisper model.

        Args:
            model_name: Model identifier (tiny, base, small, medium, large-v3).
            device: 'cpu' or 'cuda'.

        Raises:
            ModelLoadError: If whisper is not installed or the model fails to load.
        """
        try:
            import whisper  # type: ignore[import]

            logger.info("Loading OpenAI Whisper model | model=%s device=%s", model_name, device)
            self._model = whisper.load_model(model_name, device=device)
            logger.info("OpenAI Whisper model loaded successfully")
        except ImportError as exc:
            raise ModelLoadError(
                model_name,
                "Neither WhisperX nor openai-whisper is installed. "
                "Run: pip install openai-whisper",
            ) from exc
        except Exception as exc:
            raise ModelLoadError(model_name, str(exc)) from exc

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        language: str | None = None,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Transcribe one audio chunk using OpenAI Whisper.

        Args:
            audio: Float32 PCM array at 16 kHz mono.
            language: Force language or None for auto-detection.

        Returns:
            (detected_language, full_text, segments)
        """
        options: Dict[str, Any] = {
            "fp16": False,  # Disable FP16 on CPU; it causes NaN errors on non-CUDA devices
            "verbose": False,
        }
        if language:
            options["language"] = language

        result = self._model.transcribe(audio, **options)

        detected_language = result.get("language", "unknown")
        segments_raw = result.get("segments", [])
        segments = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
            }
            for seg in segments_raw
        ]
        full_text = result.get("text", "").strip()
        return detected_language, full_text, segments


# --------------------------------------------------------------------------- #
# Transcription service (orchestrator)
# --------------------------------------------------------------------------- #


class TranscriptionService:
    """
    Orchestrates chunk-level transcription and merges results into a single response.

    This class is the single entry point for transcription. It is responsible for:
    1. Delegating single chunks to the appropriate backend.
    2. Merging multi-chunk results with corrected timestamps.
    3. Deduplicating overlap regions between adjacent chunks.
    """

    def __init__(self, backend: WhisperXBackend | OpenAIWhisperBackend, settings: Settings) -> None:
        """
        Args:
            backend: Loaded transcription backend instance.
            settings: Application settings (language, task, etc.).
        """
        self._backend = backend
        self._settings = settings
        self._chunker = AudioChunker(settings)

    def transcribe(self, wav_path: "Path") -> Dict[str, Any]:
        """
        Full transcription pipeline: load audio → chunk → transcribe → merge.

        Args:
            wav_path: Path to a normalized 16 kHz mono WAV file.

        Returns:
            Dict with keys: language, duration, transcription, segments.

        Raises:
            TranscriptionError: If transcription fails for any chunk.
        """
        from pathlib import Path as _Path

        # Load the full audio into memory as a float32 array.
        # For very large files, a streaming approach (processing chunks from
        # disk without loading the full array) would be preferable, but adds
        # significant complexity. The current approach supports files up to
        # ~2 hours at 16kHz mono without exceeding 1 GB RAM.
        audio_data = self._chunker.load_audio(wav_path)
        total_duration = len(audio_data) / self._settings.audio_sample_rate

        chunks = self._chunker.chunk(audio_data)
        forced_language = self._settings.whisper_language or None

        all_segments: List[SegmentModel] = []
        detected_language = "unknown"

        with log_execution_time(logger, "transcription_pipeline"):
            for chunk in chunks:
                logger.info(
                    "Transcribing chunk | index=%d start=%.2fs end=%.2fs",
                    chunk.index,
                    chunk.start_time,
                    chunk.end_time,
                )

                try:
                    lang, _, raw_segments = self._backend.transcribe_chunk(
                        chunk.audio_data,
                        language=forced_language,
                    )
                except Exception as exc:
                    raise TranscriptionError(reason=f"Chunk {chunk.index} failed: {exc}") from exc

                # Use the language detected in the first chunk as the document language.
                # Later chunks may detect a different language due to noise or silence;
                # the first chunk is the most reliable because it starts with speech.
                if chunk.index == 0:
                    detected_language = lang

                # Adjust timestamps: Whisper returns times relative to the chunk start,
                # but we need times relative to the original audio beginning.
                adjusted = self._adjust_timestamps(raw_segments, offset=chunk.start_time)

                # Remove overlap region from all chunks except the last.
                # The overlap_seconds tail of each non-final chunk is transcribed
                # again by the next chunk with better context (words near the chunk
                # start are transcribed more accurately than words at the tail).
                if chunk.index < len(chunks) - 1:
                    adjusted = self._remove_overlap_tail(
                        adjusted,
                        chunk_end=chunk.end_time,
                        overlap=self._settings.chunk_overlap_seconds,
                    )

                all_segments.extend(adjusted)

        # Deduplicate segments that may have leaked across adjacent chunks.
        all_segments = self._deduplicate_segments(all_segments)

        full_text = " ".join(seg.text for seg in all_segments if seg.text)

        return {
            "language": detected_language,
            "duration": round(total_duration, 3),
            "transcription": full_text,
            "segments": [seg.model_dump() for seg in all_segments],
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _adjust_timestamps(
        segments: List[Dict[str, Any]],
        offset: float,
    ) -> List[SegmentModel]:
        """
        Shift segment timestamps by a time offset.

        Whisper reports timestamps relative to the audio it received (i.e.,
        the chunk). We must add the chunk's start time to convert them back
        to positions in the original, full-length audio.

        Args:
            segments: Raw segment dicts from the backend.
            offset: Seconds to add to start and end of each segment.

        Returns:
            List of SegmentModel with adjusted timestamps.
        """
        return [
            SegmentModel(
                start=round(seg["start"] + offset, 3),
                end=round(seg["end"] + offset, 3),
                text=seg["text"].strip(),
            )
            for seg in segments
            if seg.get("text", "").strip()
        ]

    @staticmethod
    def _remove_overlap_tail(
        segments: List[SegmentModel],
        chunk_end: float,
        overlap: float,
    ) -> List[SegmentModel]:
        """
        Drop segments that fall entirely within the overlap region at the chunk tail.

        The overlap region (last `overlap` seconds of a chunk) is transcribed
        again by the following chunk, which has those words at the start of its
        context window where Whisper is most accurate. We therefore discard
        the tail copy to avoid duplication.

        We only drop segments whose START time is within the overlap tail.
        Segments that START before the overlap but END within it are kept
        (partial words are better than missing words).

        Args:
            segments: Adjusted segments for this chunk.
            chunk_end: The chunk's end time in the original audio (seconds).
            overlap: Overlap duration in seconds.

        Returns:
            Segments with tail overlap removed.
        """
        cutoff = chunk_end - overlap
        return [seg for seg in segments if seg.start < cutoff]

    @staticmethod
    def _deduplicate_segments(segments: List[SegmentModel]) -> List[SegmentModel]:
        """
        Remove consecutive duplicate segments (same text, nearly same timing).

        In rare cases, boundary segments from adjacent chunks can produce
        identical text entries. We deduplicate conservatively: only remove
        a segment if its text exactly matches the preceding segment AND
        their start times are within 1 second of each other.

        Args:
            segments: All segments, potentially with duplicates.

        Returns:
            Deduplicated, ordered segment list.
        """
        if not segments:
            return segments

        deduplicated = [segments[0]]
        for current in segments[1:]:
            previous = deduplicated[-1]
            is_duplicate = (
                current.text.lower() == previous.text.lower()
                and abs(current.start - previous.start) < 1.0
            )
            if not is_duplicate:
                deduplicated.append(current)

        removed = len(segments) - len(deduplicated)
        if removed:
            logger.debug("Removed %d duplicate boundary segment(s)", removed)

        return deduplicated


# --------------------------------------------------------------------------- #
# Factory function
# --------------------------------------------------------------------------- #


def load_backend(settings: Settings) -> WhisperXBackend | OpenAIWhisperBackend:
    """
    Attempt to load WhisperX; fall back to OpenAI Whisper on ImportError.

    This function implements the progressive enhancement pattern: the service
    works correctly with the basic backend and improves automatically when
    the advanced backend is available — no configuration change required.

    Args:
        settings: Application settings.

    Returns:
        Loaded backend instance.

    Raises:
        ModelLoadError: If neither backend can be loaded.
    """
    # Try WhisperX first
    try:
        return WhisperXBackend(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    except ModelLoadError as whisperx_error:
        logger.warning(
            "WhisperX unavailable, falling back to OpenAI Whisper | reason=%s",
            whisperx_error.message,
        )

    # Fall back to OpenAI Whisper
    return OpenAIWhisperBackend(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
    )


def create_transcription_service(settings: Settings) -> TranscriptionService:
    """
    Build a fully-initialized TranscriptionService.

    Intended to be called once at application startup and the result
    stored as application state (see main.py lifespan handler).

    Args:
        settings: Application settings.

    Returns:
        Ready-to-use TranscriptionService instance.
    """
    backend = load_backend(settings)
    return TranscriptionService(backend=backend, settings=settings)
