"""
Audio normalization service using FFmpeg.

WHY FFMPEG?
-----------
FFmpeg is the de-facto standard for audio/video processing. It handles
virtually every codec and container format with a single binary, avoids
the need for per-format Python libraries, and is maintained by a large
open-source community. The subprocess approach (rather than a Python
binding like pydub) gives us direct access to all FFmpeg flags and makes
debugging straightforward — we can replay the exact command that failed.

WHY 16 kHz MONO WAV?
--------------------
Whisper was trained on audio resampled to 16 kHz mono. Feeding it any
other sample rate causes internal resampling that adds inconsistency.
Normalizing before inference ensures every transcription call receives
exactly the format the model expects, maximising accuracy and eliminating
a common source of 'the model works on my laptop but not in production'
bugs caused by mismatched input formats.

WHY PCM (pcm_s16le)?
--------------------
PCM is uncompressed, meaning numpy can memory-map the file directly
without a decoding step. This reduces CPU usage during the transcription
phase, which is already the bottleneck.
"""

import shutil
import subprocess
from pathlib import Path

from app.config.settings import Settings
from app.utils.exceptions import AudioNormalizationError, FFmpegNotFoundError
from app.utils.file_helpers import generate_unique_filename
from app.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class AudioNormalizer:
    """
    Converts any supported audio format to 16 kHz mono PCM WAV using FFmpeg.

    The normalizer is the second stage of the pipeline after validation.
    It is responsible for translating diverse input formats into the single
    canonical format that all downstream components (chunker, transcriber)
    can rely on without format-specific branching.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Application settings (sample rate, channels, codec, etc.).

        Raises:
            FFmpegNotFoundError: If ffmpeg binary is not on PATH. We raise at
                instantiation time so the error appears at startup, not mid-request.
        """
        self._settings = settings
        self._ffmpeg_path = self._locate_ffmpeg()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def normalize(self, input_path: Path, output_dir: Path) -> Path:
        """
        Normalize audio to 16 kHz mono PCM WAV.

        The output filename is derived from the input but given a '_normalized'
        suffix so both files can coexist in the same directory during debugging.

        Args:
            input_path: Path to the raw uploaded audio file.
            output_dir: Directory to write the normalized WAV file to.

        Returns:
            Path to the normalized WAV file.

        Raises:
            AudioNormalizationError: If FFmpeg returns a non-zero exit code,
                indicating that the file could not be decoded (corrupt, truncated,
                or unsupported codec variant).
        """
        output_filename = generate_unique_filename(input_path.name, suffix="_normalized")
        # Force .wav extension regardless of input extension
        output_path = output_dir / (Path(output_filename).stem + ".wav")

        with log_execution_time(logger, f"ffmpeg_normalize:{input_path.name}"):
            self._run_ffmpeg(input_path, output_path)

        logger.info(
            "Audio normalized | input=%s output=%s",
            input_path.name,
            output_path.name,
        )
        return output_path

    def get_duration(self, audio_path: Path) -> float:
        """
        Probe audio duration using ffprobe (ships with ffmpeg).

        We use ffprobe rather than reading WAV headers manually because
        it works on all formats (not just WAV) and is more reliable for
        edge cases like VBR MP3 files with inaccurate header duration.

        Args:
            audio_path: Path to any audio file.

        Returns:
            Duration in seconds as a float.

        Raises:
            AudioNormalizationError: If ffprobe cannot read the file.
        """
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            # ffprobe ships alongside ffmpeg; if it is missing, something is very wrong.
            logger.warning("ffprobe not found, attempting duration from WAV header")
            return self._duration_from_wav_header(audio_path)

        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_entries", "format=duration",
            str(audio_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise AudioNormalizationError(
                filename=audio_path.name,
                reason=f"ffprobe failed: {exc.stderr.strip()}",
            ) from exc

        import json
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0.0))
        logger.debug("Probed duration | file=%s duration=%.2fs", audio_path.name, duration)
        return duration

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _locate_ffmpeg(self) -> str:
        """
        Find the ffmpeg binary on PATH.

        Raises:
            FFmpegNotFoundError: If ffmpeg is not installed.
        """
        path = shutil.which("ffmpeg")
        if not path:
            raise FFmpegNotFoundError()
        logger.info("FFmpeg located | path=%s", path)
        return path

    def _run_ffmpeg(self, input_path: Path, output_path: Path) -> None:
        """
        Execute the FFmpeg normalization command.

        Command breakdown:
          -y                  Overwrite output without prompting (non-interactive).
          -i <input>          Input file.
          -vn                 Drop any video stream (e.g. MP4 with cover art).
          -acodec pcm_s16le   16-bit signed little-endian PCM — the WAV standard.
          -ar 16000           Resample to 16 kHz (Whisper training rate).
          -ac 1               Downmix to mono. Stereo adds no value for speech
                              transcription and doubles memory consumption.
          <output>            Output WAV file.

        Args:
            input_path: Source audio file.
            output_path: Destination WAV file.

        Raises:
            AudioNormalizationError: On non-zero FFmpeg exit code.
        """
        cmd = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-vn",                             # Strip video streams
            "-acodec", self._settings.audio_bit_depth,
            "-ar", str(self._settings.audio_sample_rate),
            "-ac", str(self._settings.audio_channels),
            str(output_path),
        ]

        logger.debug("Running FFmpeg | cmd=%s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout for very large files
                check=False,   # We inspect returncode manually for better error messages
            )
        except subprocess.TimeoutExpired:
            raise AudioNormalizationError(
                filename=input_path.name,
                reason="FFmpeg timed out after 300 seconds",
            )
        except FileNotFoundError as exc:
            raise AudioNormalizationError(
                filename=input_path.name,
                reason=f"FFmpeg binary not found: {exc}",
            ) from exc

        if result.returncode != 0:
            # FFmpeg writes diagnostics to stderr; include the last 500 chars
            # to surface the actionable error line without flooding the log.
            stderr_tail = result.stderr[-500:].strip()
            raise AudioNormalizationError(
                filename=input_path.name,
                reason=f"FFmpeg exited with code {result.returncode}: {stderr_tail}",
            )

    def _duration_from_wav_header(self, wav_path: Path) -> float:
        """
        Fallback duration calculation by reading the WAV file header.

        This is used only when ffprobe is unavailable. It works only for
        properly-formed WAV files; use ffprobe for other formats.

        Args:
            wav_path: Path to a WAV file.

        Returns:
            Duration in seconds.
        """
        import wave
        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception as exc:
            logger.warning("Could not read WAV header for duration | error=%s", exc)
            return 0.0
