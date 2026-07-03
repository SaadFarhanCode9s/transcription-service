"""
Tests for the upload validation service.

These tests verify each validation rule in isolation, using real files
written to tmp_path rather than mocking filesystem calls — this gives
us confidence that the validation logic works end-to-end without relying
on implementation details.
"""

import io
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.validator import UploadValidator
from app.utils.exceptions import FileMissingError, FileTooLargeError, UnsupportedFormatError
from tests.conftest import generate_wav_bytes


class TestFileMissingValidation:
    """Presence checks: filename absent or empty string."""

    def test_raises_when_filename_is_none(self, tmp_path: Path, validator: UploadValidator) -> None:
        """None filename (multipart field without filename attribute) must be rejected."""
        dummy_file = tmp_path / "dummy.wav"
        dummy_file.write_bytes(b"\x00" * 100)

        with pytest.raises(FileMissingError) as exc_info:
            validator.validate(filename=None, file_path=dummy_file, file_size=100)

        assert exc_info.value.code == "file_missing"

    def test_raises_when_filename_is_empty_string(self, tmp_path: Path, validator: UploadValidator) -> None:
        """Empty string filename must also be rejected."""
        dummy_file = tmp_path / "dummy.wav"
        dummy_file.write_bytes(b"\x00" * 100)

        with pytest.raises(FileMissingError):
            validator.validate(filename="", file_path=dummy_file, file_size=100)

    def test_raises_when_filename_is_whitespace(self, tmp_path: Path, validator: UploadValidator) -> None:
        """Whitespace-only filename should be treated as missing."""
        dummy_file = tmp_path / "dummy.wav"
        dummy_file.write_bytes(b"\x00" * 100)

        with pytest.raises(FileMissingError):
            validator.validate(filename="   ", file_path=dummy_file, file_size=100)


class TestExtensionValidation:
    """Extension allow-list checks."""

    @pytest.mark.parametrize("extension", ["wav", "mp3", "flac", "m4a", "ogg", "aac"])
    def test_accepts_all_supported_extensions(
        self, tmp_path: Path, validator: UploadValidator, extension: str
    ) -> None:
        """All six supported formats should pass extension check."""
        # Write real WAV magic bytes so MIME check passes too
        wav_bytes = generate_wav_bytes(0.1)
        audio_file = tmp_path / f"audio.{extension}"
        audio_file.write_bytes(wav_bytes)

        # Should not raise (MIME check may warn but won't block on valid WAV bytes
        # for wav/mp3/etc — we use wav bytes as a stand-in for this test)
        # We only care that extension check passes here; MIME check may log a warning.
        try:
            validator._check_extension(f"audio.{extension}")
        except UnsupportedFormatError:
            pytest.fail(f"Extension '{extension}' should be accepted but was rejected")

    def test_rejects_unsupported_extension(self, tmp_path: Path, validator: UploadValidator) -> None:
        """Extensions not in the allow-list must produce UnsupportedFormatError."""
        dummy_file = tmp_path / "audio.xyz"
        dummy_file.write_bytes(b"\x00" * 100)

        with pytest.raises(UnsupportedFormatError) as exc_info:
            validator.validate(filename="audio.xyz", file_path=dummy_file, file_size=100)

        assert exc_info.value.code == "unsupported_format"
        assert "xyz" in exc_info.value.message.lower() or "audio.xyz" in exc_info.value.message

    @pytest.mark.parametrize("filename", ["audio.PDF", "audio.EXE", "audio.TXT"])
    def test_rejects_uppercase_unsupported_extension(
        self, tmp_path: Path, validator: UploadValidator, filename: str
    ) -> None:
        """Extension check must be case-insensitive and still reject non-audio."""
        dummy_file = tmp_path / filename
        dummy_file.write_bytes(b"\x00" * 100)

        with pytest.raises(UnsupportedFormatError):
            validator.validate(filename=filename, file_path=dummy_file, file_size=100)


class TestSizeValidation:
    """File size limit checks."""

    def test_accepts_file_within_size_limit(self, tmp_path: Path, validator: UploadValidator) -> None:
        """Files under the configured limit must pass size validation."""
        wav_bytes = generate_wav_bytes(0.1)
        audio_file = tmp_path / "small.wav"
        audio_file.write_bytes(wav_bytes)

        # Should not raise
        validator._check_size(len(wav_bytes))

    def test_rejects_file_exceeding_size_limit(self, tmp_path: Path, validator: UploadValidator) -> None:
        """Files exceeding the configured limit must raise FileTooLargeError."""
        # test_settings has max 10 MB; simulate 11 MB
        oversized_bytes = 11 * 1024 * 1024

        with pytest.raises(FileTooLargeError) as exc_info:
            validator._check_size(oversized_bytes)

        assert exc_info.value.code == "file_too_large"
        assert exc_info.value.details["size_bytes"] == oversized_bytes

    def test_accepts_file_exactly_at_limit(self, tmp_path: Path, validator: UploadValidator) -> None:
        """A file exactly at the size limit must pass."""
        from app.config.settings import Settings
        settings = validator._settings
        max_bytes = settings.max_upload_size_bytes

        # Must not raise — equal to limit is allowed
        validator._check_size(max_bytes)

    def test_rejects_file_one_byte_over_limit(self, tmp_path: Path, validator: UploadValidator) -> None:
        """One byte over the limit must trigger rejection."""
        settings = validator._settings
        over_limit = settings.max_upload_size_bytes + 1

        with pytest.raises(FileTooLargeError):
            validator._check_size(over_limit)


class TestFullValidation:
    """Integration-style tests for the full validate() method."""

    def test_valid_wav_passes_all_checks(self, tmp_path: Path, validator: UploadValidator) -> None:
        """A real WAV file with correct name should pass all validation."""
        wav_bytes = generate_wav_bytes(1.0)
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(wav_bytes)

        # Should not raise
        validator.validate(
            filename="audio.wav",
            file_path=audio_file,
            file_size=len(wav_bytes),
        )
