"""
Tests for LocalFileStorage.

These tests verify path provisioning and JSON persistence without
touching the transcription or normalization layers.
"""

import json
from pathlib import Path

import pytest

from app.services.storage import LocalFileStorage


class TestLocalFileStorage:
    """Unit tests for LocalFileStorage."""

    def test_upload_path_is_within_upload_dir(self, storage: LocalFileStorage, tmp_path: Path) -> None:
        """get_upload_path must return a path inside the uploads directory."""
        path = storage.get_upload_path("audio_123.wav")
        assert path.parent == Path(storage._upload_dir)

    def test_upload_path_includes_filename(self, storage: LocalFileStorage) -> None:
        """The returned path must end with the provided filename."""
        path = storage.get_upload_path("test_file.wav")
        assert path.name == "test_file.wav"

    def test_output_path_is_within_output_dir(self, storage: LocalFileStorage) -> None:
        """get_output_path must return a path inside the output directory."""
        path = storage.get_output_path("result_123")
        assert path.parent == Path(storage._output_dir)

    def test_output_path_has_json_extension(self, storage: LocalFileStorage) -> None:
        """Output path must have .json extension."""
        path = storage.get_output_path("result_123")
        assert path.suffix == ".json"

    def test_save_result_creates_file(self, storage: LocalFileStorage) -> None:
        """save_result must write a JSON file to the output directory."""
        result = {
            "language": "en",
            "duration": 5.0,
            "transcription": "Hello.",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello."}],
        }
        saved_path = storage.save_result(stem="test_result", result=result)

        assert saved_path.exists()
        assert saved_path.suffix == ".json"

    def test_save_result_json_is_valid(self, storage: LocalFileStorage) -> None:
        """The saved file must contain valid JSON matching the input dict."""
        result = {"language": "fr", "duration": 10.0, "transcription": "Bonjour.", "segments": []}
        saved_path = storage.save_result(stem="french_test", result=result)

        with open(saved_path) as f:
            loaded = json.load(f)

        assert loaded["language"] == "fr"
        assert loaded["duration"] == 10.0

    def test_save_result_handles_unicode(self, storage: LocalFileStorage) -> None:
        """Unicode text (Chinese, Arabic, emoji) must be preserved correctly."""
        result = {
            "language": "zh",
            "duration": 3.0,
            "transcription": "你好世界 مرحبا بالعالم 🎤",
            "segments": [],
        }
        saved_path = storage.save_result(stem="unicode_test", result=result)

        with open(saved_path, encoding="utf-8") as f:
            content = f.read()

        assert "你好世界" in content
        assert "🎤" in content

    def test_cleanup_upload_deletes_existing_file(self, storage: LocalFileStorage, tmp_path: Path) -> None:
        """cleanup_upload must delete files that exist."""
        temp_file = tmp_path / "to_delete.wav"
        temp_file.write_bytes(b"data")
        assert temp_file.exists()

        storage.cleanup_upload(temp_file)
        assert not temp_file.exists()

    def test_cleanup_upload_is_safe_for_missing_files(self, storage: LocalFileStorage, tmp_path: Path) -> None:
        """cleanup_upload must not raise if the file does not exist."""
        non_existent = tmp_path / "ghost.wav"
        # Must not raise
        storage.cleanup_upload(non_existent)

    def test_cleanup_upload_handles_multiple_paths(self, storage: LocalFileStorage, tmp_path: Path) -> None:
        """cleanup_upload must delete all provided paths."""
        files = [tmp_path / f"file{i}.wav" for i in range(3)]
        for f in files:
            f.write_bytes(b"data")

        storage.cleanup_upload(*files)

        for f in files:
            assert not f.exists()

    def test_directories_are_created_on_init(self, test_settings) -> None:
        """Storage init must create upload and output directories if they don't exist."""
        # Instantiating LocalFileStorage triggers ensure_directory for both dirs.
        # The test_settings fixture uses a fresh tmp_path, so these dirs do not
        # exist before this point — Storage creation must create them.
        new_storage = LocalFileStorage(test_settings)

        assert Path(test_settings.upload_dir).exists()
        assert Path(test_settings.output_dir).exists()
