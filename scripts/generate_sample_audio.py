"""
Script to generate a sample WAV audio file for manual testing.

Running this script creates `sample_audio/sample_speech.wav` which can be
used with curl to test the /transcribe endpoint without finding a real
audio file.

The generated audio contains a 440 Hz sine wave (a musical 'A' note),
which Whisper will attempt to transcribe as silence or noise — this is
expected. For real transcription testing, use an actual speech recording.

Usage:
    python scripts/generate_sample_audio.py
"""

import math
import struct
import sys
import wave
from pathlib import Path


def generate_sine_wave(
    frequency_hz: float,
    duration_seconds: float,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> bytes:
    """
    Generate a sine wave as 16-bit PCM bytes.

    Args:
        frequency_hz: Sine wave frequency in Hz.
        duration_seconds: Duration of the audio.
        sample_rate: Sample rate in Hz.
        amplitude: Wave amplitude (0.0–1.0, where 1.0 would clip).

    Returns:
        Raw 16-bit PCM audio bytes.
    """
    n_samples = int(duration_seconds * sample_rate)
    raw = []
    for i in range(n_samples):
        t = i / sample_rate
        sample_value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency_hz * t))
        raw.append(struct.pack("<h", sample_value))  # little-endian 16-bit signed int
    return b"".join(raw)


def write_wav(output_path: Path, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
    """
    Write PCM bytes to a WAV file.

    Args:
        output_path: Destination file path.
        pcm_bytes: Raw 16-bit mono PCM data.
        sample_rate: Sample rate used to generate pcm_bytes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)    # Mono
        wf.setsampwidth(2)    # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def main() -> None:
    output_path = Path(__file__).parent.parent / "sample_audio" / "sample_speech.wav"

    print(f"Generating sample audio → {output_path}")

    # 3-second tone — long enough for Whisper to process, short enough to be quick
    pcm = generate_sine_wave(frequency_hz=440.0, duration_seconds=3.0)
    write_wav(output_path, pcm)

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Written {size_kb:.1f} KB WAV file")
    print(f"\nTest with:")
    print(f"  curl -X POST http://localhost:8000/api/v1/transcribe \\")
    print(f"       -F 'file=@{output_path}'")


if __name__ == "__main__":
    main()
