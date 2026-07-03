"""
Audio chunking service for long audio files.

WHY CHUNK AUDIO?
----------------
Whisper has a fixed 30-second context window. Audio longer than 30 seconds
is processed internally by Whisper using a sliding window, but this can
produce boundary artefacts: words that straddle a 30-second mark may be
dropped, repeated, or garbled because neither window has enough context to
transcribe them accurately.

More importantly, loading a 2-hour WAV file into numpy as a single array
requires ~230 MB of RAM (2h × 16000 samples/s × 2 bytes/sample). On
modest hardware this can exhaust memory, especially when multiple requests
are processed concurrently.

By chunking at the Python level:
1. Each chunk fits comfortably in the Whisper context window.
2. Memory usage stays bounded (chunk_duration × sample_rate × 2 bytes).
3. We can parallelize chunk transcription across CPU cores or GPU workers.
4. Failed chunks can be retried independently without reprocessing the file.

OVERLAP STRATEGY
----------------
A 2-second overlap between adjacent chunks prevents words at the boundary
from being cut mid-utterance. The overlap region is transcribed twice
(once per neighboring chunk); we deduplicate by taking the segment from
whichever chunk provides a cleaner transcription (i.e., the chunk where
the word appears earlier in the context window, away from the boundary).

In practice, we simply truncate the overlap from the right side of each
non-final chunk's transcript, since words early in a chunk are always
transcribed more accurately than words at the tail.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from app.config.settings import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioChunk:
    """
    Represents a slice of audio data ready for transcription.

    Attributes:
        index: Zero-based position of this chunk in the sequence.
        audio_data: Numpy array of float32 PCM samples (16 kHz mono).
        start_time: Offset in seconds from the beginning of the original audio.
        end_time: End offset in seconds.
    """

    index: int
    audio_data: np.ndarray
    start_time: float
    end_time: float

    def __len__(self) -> int:
        """Return number of audio samples in this chunk."""
        return len(self.audio_data)


class AudioChunker:
    """
    Splits a normalized WAV file into overlapping fixed-length chunks.

    The chunker operates on numpy arrays (not files) to avoid repeated
    disk I/O. The caller is responsible for loading the WAV into numpy
    before passing it to `chunk()`.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Application settings with chunk_duration_seconds,
                      chunk_overlap_seconds, and audio_sample_rate.
        """
        self._settings = settings
        self._sample_rate = settings.audio_sample_rate
        self._chunk_samples = settings.chunk_duration_seconds * settings.audio_sample_rate
        self._overlap_samples = settings.chunk_overlap_seconds * settings.audio_sample_rate

    def chunk(self, audio_data: np.ndarray) -> List[AudioChunk]:
        """
        Divide audio into overlapping chunks.

        Args:
            audio_data: Float32 numpy array of PCM samples at the configured
                        sample rate. Must be 1-D (mono).

        Returns:
            List of AudioChunk objects in order. For audio shorter than
            chunk_duration_seconds, returns a single chunk.
        """
        total_samples = len(audio_data)
        total_duration = total_samples / self._sample_rate

        if total_samples <= self._chunk_samples:
            # Short audio: skip chunking overhead entirely.
            logger.debug(
                "Audio shorter than chunk size, processing as single chunk | duration=%.2fs",
                total_duration,
            )
            return [AudioChunk(index=0, audio_data=audio_data, start_time=0.0, end_time=total_duration)]

        chunks: List[AudioChunk] = []

        # Step size = chunk_size - overlap ensures consecutive chunks share
        # `overlap_samples` samples at their boundaries.
        step_samples = self._chunk_samples - self._overlap_samples

        logger.info(
            "Chunking audio | total_duration=%.2fs chunk_size=%ds overlap=%ds expected_chunks=%d",
            total_duration,
            self._settings.chunk_duration_seconds,
            self._settings.chunk_overlap_seconds,
            self._estimate_chunk_count(total_samples),
        )

        chunk_index = 0
        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(start_sample + self._chunk_samples, total_samples)

            # Convert sample positions to seconds for timestamp tracking.
            start_time = start_sample / self._sample_rate
            end_time = end_sample / self._sample_rate

            chunk_audio = audio_data[start_sample:end_sample]
            chunks.append(
                AudioChunk(
                    index=chunk_index,
                    audio_data=chunk_audio,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

            logger.debug(
                "Created chunk | index=%d start=%.2fs end=%.2fs samples=%d",
                chunk_index,
                start_time,
                end_time,
                len(chunk_audio),
            )

            chunk_index += 1
            start_sample += step_samples

        logger.info("Chunking complete | total_chunks=%d", len(chunks))
        return chunks

    def load_audio(self, wav_path: Path) -> np.ndarray:
        """
        Load a normalized WAV file into a float32 numpy array.

        We normalize to float32 in [-1.0, 1.0] because that is the format
        Whisper's mel-spectrogram computation expects. Loading as int16 and
        converting avoids the quality loss of decode → re-encode cycles.

        Args:
            wav_path: Path to a 16 kHz mono PCM WAV file.

        Returns:
            Float32 numpy array of audio samples.

        Raises:
            OSError: If the file cannot be read.
        """
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        # int16 PCM → float32 in [-1.0, 1.0]
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

    def _estimate_chunk_count(self, total_samples: int) -> int:
        """
        Estimate how many chunks will be produced without running the full loop.

        Used only for log messages to give operators early visibility into
        how long a transcription job might take.

        Args:
            total_samples: Total number of audio samples.

        Returns:
            Estimated chunk count (may be off by ±1 at boundaries).
        """
        step = self._chunk_samples - self._overlap_samples
        return max(1, int(np.ceil((total_samples - self._overlap_samples) / step)))
