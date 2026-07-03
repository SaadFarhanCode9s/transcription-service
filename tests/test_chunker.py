"""
Tests for the audio chunking service.

These tests verify chunking behavior using synthetic numpy arrays —
no real audio files or FFmpeg needed. The chunker is a pure data
transformation and should be trivially testable in isolation.
"""

from pathlib import Path

import numpy as np
import pytest

from app.config.settings import Settings
from app.services.audio_chunker import AudioChunk, AudioChunker


SAMPLE_RATE = 16000  # 16 kHz, same as production setting


def make_audio(seconds: float) -> np.ndarray:
    """Generate a float32 sine wave array of the given duration."""
    n = int(seconds * SAMPLE_RATE)
    t = np.linspace(0, seconds, n, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


class TestChunking:
    """Verify that audio is split into correctly-sized, correctly-timed chunks."""

    def test_short_audio_returns_single_chunk(self, chunker: AudioChunker) -> None:
        """Audio shorter than chunk_duration must not be split."""
        audio = make_audio(5.0)  # 5 seconds < 30-second chunk size
        chunks = chunker.chunk(audio)

        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].start_time == 0.0
        assert np.allclose(chunks[0].end_time, 5.0, atol=0.01)

    def test_audio_exactly_at_chunk_boundary(self, chunker: AudioChunker) -> None:
        """Audio exactly equal to chunk_duration should produce one chunk."""
        audio = make_audio(30.0)  # Exactly the chunk size
        chunks = chunker.chunk(audio)

        assert len(chunks) == 1

    def test_long_audio_produces_multiple_chunks(self, chunker: AudioChunker) -> None:
        """Audio longer than chunk_duration must produce multiple chunks."""
        audio = make_audio(65.0)  # 65 seconds → expect at least 2 chunks
        chunks = chunker.chunk(audio)

        assert len(chunks) >= 2

    def test_chunk_indices_are_sequential(self, chunker: AudioChunker) -> None:
        """Chunk indices must be 0, 1, 2, ... with no gaps."""
        audio = make_audio(90.0)
        chunks = chunker.chunk(audio)

        for expected_idx, chunk in enumerate(chunks):
            assert chunk.index == expected_idx

    def test_chunk_timestamps_are_non_decreasing(self, chunker: AudioChunker) -> None:
        """Start times must increase monotonically across chunks."""
        audio = make_audio(90.0)
        chunks = chunker.chunk(audio)

        for i in range(1, len(chunks)):
            assert chunks[i].start_time > chunks[i - 1].start_time, (
                f"Chunk {i} start ({chunks[i].start_time}) should be after "
                f"chunk {i-1} start ({chunks[i - 1].start_time})"
            )

    def test_first_chunk_starts_at_zero(self, chunker: AudioChunker) -> None:
        """The first chunk must start at t=0."""
        audio = make_audio(45.0)
        chunks = chunker.chunk(audio)

        assert chunks[0].start_time == 0.0

    def test_last_chunk_covers_audio_end(self, chunker: AudioChunker) -> None:
        """The last chunk's end time must equal the audio duration."""
        duration = 65.0
        audio = make_audio(duration)
        chunks = chunker.chunk(audio)

        assert np.allclose(chunks[-1].end_time, duration, atol=0.1)

    def test_overlap_means_adjacent_chunks_share_samples(self, chunker: AudioChunker) -> None:
        """
        With overlap, consecutive chunks must share samples at their boundary.

        Concretely: the end time of chunk N should be >= start time of chunk N+1.
        This confirms the overlap window exists and words at boundaries are
        covered by both chunks.
        """
        audio = make_audio(70.0)
        chunks = chunker.chunk(audio)

        for i in range(len(chunks) - 1):
            assert chunks[i].end_time > chunks[i + 1].start_time, (
                f"Chunk {i} end ({chunks[i].end_time}) should overlap "
                f"chunk {i+1} start ({chunks[i+1].start_time})"
            )

    def test_chunk_audio_data_is_not_empty(self, chunker: AudioChunker) -> None:
        """Every chunk must contain at least one audio sample."""
        audio = make_audio(60.0)
        chunks = chunker.chunk(audio)

        for chunk in chunks:
            assert len(chunk.audio_data) > 0

    def test_all_samples_covered(self, chunker: AudioChunker) -> None:
        """
        Due to overlap, the union of all chunk samples should cover the full audio.

        We verify this by checking that every sample index in the original audio
        appears in at least one chunk. For large arrays, we sample 1000 random indices.
        """
        audio = make_audio(90.0)
        chunks = chunker.chunk(audio)

        total_samples = len(audio)
        sample_rate = chunker._sample_rate

        # Rebuild which sample indices each chunk covers
        covered = set()
        for chunk in chunks:
            start_idx = int(chunk.start_time * sample_rate)
            end_idx = start_idx + len(chunk.audio_data)
            covered.update(range(start_idx, min(end_idx, total_samples)))

        # Sample 1000 random indices and verify coverage
        rng = np.random.default_rng(42)
        test_indices = rng.integers(0, total_samples, size=min(1000, total_samples))
        uncovered = [i for i in test_indices if i not in covered]
        assert len(uncovered) == 0, f"{len(uncovered)} sample positions not covered by any chunk"


class TestChunkDataclass:
    """Tests for the AudioChunk dataclass."""

    def test_len_returns_sample_count(self) -> None:
        """len(chunk) must return the number of audio samples."""
        audio = np.zeros(1600, dtype=np.float32)
        chunk = AudioChunk(index=0, audio_data=audio, start_time=0.0, end_time=0.1)
        assert len(chunk) == 1600
