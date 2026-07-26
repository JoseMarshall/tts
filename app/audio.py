"""Audio helpers: float <-> PCM conversion and (streaming) WAV framing."""
from __future__ import annotations

import struct

import numpy as np

# Sentinel used for the size fields of a WAV that is written incrementally,
# before the total length is known. Most players tolerate this for streams.
_STREAM_SIZE = 0xFFFFFFFF


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float32 audio in [-1, 1] to little-endian 16-bit PCM bytes."""
    if samples.dtype != np.float32 and samples.dtype != np.float64:
        # Already integer PCM — assume int16.
        return np.asarray(samples, dtype="<i2").tobytes()
    clipped = np.clip(samples, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    return ints.tobytes()


def wav_header(sample_rate: int, channels: int = 1, bits: int = 16,
               data_size: int | None = None) -> bytes:
    """Build a 44-byte WAV/RIFF header.

    Pass ``data_size`` (bytes of PCM) for a complete file, or leave it ``None``
    to emit a streaming header whose size fields are left "open".
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    if data_size is None:
        riff_size = _STREAM_SIZE
        chunk_size = _STREAM_SIZE
    else:
        riff_size = 36 + data_size
        chunk_size = data_size
    return b"".join([
        b"RIFF",
        struct.pack("<I", riff_size),
        b"WAVE",
        b"fmt ",
        struct.pack("<I", 16),           # PCM fmt chunk size
        struct.pack("<H", 1),            # audio format: 1 = PCM
        struct.pack("<H", channels),
        struct.pack("<I", sample_rate),
        struct.pack("<I", byte_rate),
        struct.pack("<H", block_align),
        struct.pack("<H", bits),
        b"data",
        struct.pack("<I", chunk_size),
    ])


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1,
               bits: int = 16) -> bytes:
    """Wrap a complete PCM buffer in a correctly-sized WAV container."""
    return wav_header(sample_rate, channels, bits, len(pcm)) + pcm


# Map of the response formats we can emit to their HTTP content types.
CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
