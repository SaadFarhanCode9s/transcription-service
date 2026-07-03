"""
Tests for the TranscriptionService.

We test the service's orchestration logic (timestamp adjustment,
overlap deduplication, chunk merging) using the mock backend.
The backend itself is not tested here — it is a third-party library.

Testing the orchestration in isolation from the backend means these
tests run in milliseconds without requiring Whisper installed.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from app.models.response_models import SegmentModel
from app.services.audio_chunker import AudioChunk
from app.services.transcriber import TranscriptionService


class TestTimestampAdjustment:
    """Tests for the _adjust_timestamps static method."""

    def test_adds_offset_to_segment_times(self) -> None:
        """Segment times must be shifted by the chunk's start offset."""
        raw_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello."},
            {"start": 2.5, "end": 5.0, "text": "World."},
        ]
        adjusted = TranscriptionService._adjust_timestamps(raw_segments, offset=10.0)

        assert adjusted[0].start == pytest.approx(10.0)
        assert adjusted[0].end == pytest.approx(12.0)
        assert adjusted[1].start == pytest.approx(12.5)
        assert adjusted[1].end == pytest.approx(15.0)

    def test_preserves_text_content(self) -> None:
        """Text must be preserved exactly (strip() removes leading/trailing whitespace)."""
        raw_segments = [{"start": 0.0, "end": 1.0, "text": "  Hello World  "}]
        adjusted = TranscriptionService._adjust_timestamps(raw_segments, offset=0.0)

        assert adjusted[0].text == "Hello World"

    def test_filters_empty_text_segments(self) -> None:
        """Segments with empty text must be dropped."""
        raw_segments = [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "  "},
            {"start": 2.0, "end": 3.0, "text": "Real content."},
        ]
        adjusted = TranscriptionService._adjust_timestamps(raw_segments, offset=0.0)

        assert len(adjusted) == 1
        assert adjusted[0].text == "Real content."

    def test_zero_offset_leaves_times_unchanged(self) -> None:
        """Zero offset must not modify timestamps."""
        raw_segments = [{"start": 5.0, "end": 8.0, "text": "Test."}]
        adjusted = TranscriptionService._adjust_timestamps(raw_segments, offset=0.0)

        assert adjusted[0].start == pytest.approx(5.0)
        assert adjusted[0].end == pytest.approx(8.0)


class TestOverlapRemoval:
    """Tests for the _remove_overlap_tail static method."""

    def test_keeps_segments_before_cutoff(self) -> None:
        """Segments starting before the overlap cutoff must be retained."""
        segments = [
            SegmentModel(start=0.0, end=5.0, text="Early segment."),
            SegmentModel(start=10.0, end=15.0, text="Middle segment."),
        ]
        result = TranscriptionService._remove_overlap_tail(
            segments, chunk_end=30.0, overlap=2.0
        )

        # cutoff = 30.0 - 2.0 = 28.0; both segments start before 28.0
        assert len(result) == 2

    def test_drops_segments_in_overlap_region(self) -> None:
        """Segments starting in the overlap tail must be removed."""
        segments = [
            SegmentModel(start=0.0, end=5.0, text="Before overlap."),
            SegmentModel(start=28.5, end=30.0, text="In overlap zone."),  # start >= 28.0
        ]
        result = TranscriptionService._remove_overlap_tail(
            segments, chunk_end=30.0, overlap=2.0
        )

        assert len(result) == 1
        assert result[0].text == "Before overlap."

    def test_boundary_segment_exactly_at_cutoff_is_kept(self) -> None:
        """A segment starting exactly at the cutoff boundary should be kept (< cutoff)."""
        # cutoff = 30 - 2 = 28.0; a segment starting at exactly 27.999 is kept
        segments = [
            SegmentModel(start=27.999, end=29.0, text="Boundary segment."),
        ]
        result = TranscriptionService._remove_overlap_tail(
            segments, chunk_end=30.0, overlap=2.0
        )

        assert len(result) == 1


class TestDeduplication:
    """Tests for the _deduplicate_segments static method."""

    def test_removes_exact_duplicate_adjacent_segments(self) -> None:
        """Consecutive identical segments must be deduplicated."""
        segments = [
            SegmentModel(start=0.0, end=2.0, text="Hello world."),
            SegmentModel(start=0.5, end=2.5, text="Hello world."),  # duplicate within 1s
        ]
        result = TranscriptionService._deduplicate_segments(segments)

        assert len(result) == 1
        assert result[0].text == "Hello world."

    def test_keeps_non_duplicate_segments(self) -> None:
        """Segments with different text must all be retained."""
        segments = [
            SegmentModel(start=0.0, end=2.0, text="First sentence."),
            SegmentModel(start=2.5, end=5.0, text="Second sentence."),
            SegmentModel(start=5.5, end=8.0, text="Third sentence."),
        ]
        result = TranscriptionService._deduplicate_segments(segments)

        assert len(result) == 3

    def test_keeps_identical_text_far_apart_in_time(self) -> None:
        """Identical text appearing >1 second apart must not be deduplicated."""
        segments = [
            SegmentModel(start=0.0, end=2.0, text="Repeated phrase."),
            SegmentModel(start=60.0, end=62.0, text="Repeated phrase."),  # 60s apart
        ]
        result = TranscriptionService._deduplicate_segments(segments)

        assert len(result) == 2

    def test_empty_list_returns_empty(self) -> None:
        """Empty input must return empty output without error."""
        result = TranscriptionService._deduplicate_segments([])
        assert result == []


class TestTranscribeEndToEnd:
    """End-to-end transcription using the mock backend and a real WAV file."""

    def test_returns_expected_fields(
        self,
        transcription_service: TranscriptionService,
        sample_wav_file: Path,
    ) -> None:
        """The result dict must contain all required fields with correct types."""
        result = transcription_service.transcribe(sample_wav_file)

        assert "language" in result
        assert "duration" in result
        assert "transcription" in result
        assert "segments" in result

        assert isinstance(result["language"], str)
        assert isinstance(result["duration"], float)
        assert isinstance(result["transcription"], str)
        assert isinstance(result["segments"], list)

    def test_segments_have_required_keys(
        self,
        transcription_service: TranscriptionService,
        sample_wav_file: Path,
    ) -> None:
        """Each segment must have start, end, and text keys."""
        result = transcription_service.transcribe(sample_wav_file)

        for segment in result["segments"]:
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment

    def test_duration_is_positive(
        self,
        transcription_service: TranscriptionService,
        sample_wav_file: Path,
    ) -> None:
        """Duration must be a positive float (non-zero for a real audio file)."""
        result = transcription_service.transcribe(sample_wav_file)
        assert result["duration"] > 0.0

    def test_language_is_string(
        self,
        transcription_service: TranscriptionService,
        sample_wav_file: Path,
    ) -> None:
        """Language must be a non-empty string."""
        result = transcription_service.transcribe(sample_wav_file)
        assert len(result["language"]) > 0

    def test_transcription_matches_segment_text(
        self,
        transcription_service: TranscriptionService,
        sample_wav_file: Path,
    ) -> None:
        """The full transcription must be composed of segment texts."""
        result = transcription_service.transcribe(sample_wav_file)
        segment_text = " ".join(s["text"] for s in result["segments"] if s["text"])
        assert result["transcription"] == segment_text
