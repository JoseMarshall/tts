"""TTS engines.

An engine turns a :class:`~app.schemas.TTSRequest` into a stream of float32
audio chunks (mono, in [-1, 1], at ``settings.sample_rate``). Non-streaming
synthesis is just "consume the stream and concatenate", so streaming is the
only primitive an engine must implement.

Two backends are provided:

* ``MockEngine`` — a dependency-free tone generator for local development and
  for exercising the streaming/HTTP plumbing without a GPU.
* ``QwenEngine`` — wraps the ``qwen-tts`` package and the real Qwen3-TTS model.
  The single place that talks to the library is :meth:`QwenEngine._raw_stream`;
  adapt it there if the library's signature differs from what's documented.
"""
from __future__ import annotations

import abc
import logging
from typing import Iterator

import numpy as np

from .config import Settings
from .schemas import TTSRequest

log = logging.getLogger("tts.engine")


class TTSEngine(abc.ABC):
    def __init__(self, settings: Settings, model_id: str | None = None):
        self.settings = settings
        self.model_id = model_id or settings.model_id
        self.sample_rate = settings.sample_rate

    @abc.abstractmethod
    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        """Yield successive float32 mono audio chunks for ``req``."""

    def synthesize(self, req: TTSRequest) -> np.ndarray:
        """Full (non-streaming) synthesis — concatenate the stream."""
        parts = list(self.stream(req))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def warmup(self) -> None:  # optional
        pass


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #
class MockEngine(TTSEngine):
    """Generates a short, pleasant chord whose length scales with text length.

    Useful to verify streaming, backpressure and client behaviour end-to-end
    without downloading a multi-gigabyte model or needing a GPU.
    """

    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        sr = self.sample_rate
        # ~90 ms of audio per character, clamped to a sane range.
        seconds = min(max(len(req.text) * 0.09, 0.5), 30.0)
        total = int(seconds * sr)
        chunk = self.settings.stream_chunk_samples
        freqs = (220.0, 277.18, 329.63)  # A3 major-ish chord
        for start in range(0, total, chunk):
            n = min(chunk, total - start)
            t = (np.arange(start, start + n) / sr).astype(np.float32)
            wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
            # Gentle fade in/out over the whole clip to avoid clicks.
            env = np.minimum(t / 0.05, (seconds - t) / 0.05)
            env = np.clip(env, 0.0, 1.0).astype(np.float32)
            yield (0.3 * wave * env).astype(np.float32)


# --------------------------------------------------------------------------- #
# Qwen3-TTS backend
# --------------------------------------------------------------------------- #
class QwenEngine(TTSEngine):
    def __init__(self, settings: Settings, model_id: str | None = None):
        super().__init__(settings, model_id)
        import torch  # noqa: F401 (validate availability early)
        from qwen_tts import Qwen3TTSModel

        dtypes = {
            "bfloat16": __import__("torch").bfloat16,
            "float16": __import__("torch").float16,
            "float32": __import__("torch").float32,
        }
        log.info("Loading %s on %s ...", self.model_id, settings.device)
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=settings.device,
            dtype=dtypes[settings.dtype],
            attn_implementation=settings.attn_implementation,
        )
        self.sample_rate = settings.sample_rate

    def warmup(self) -> None:
        try:
            list(self.stream(TTSRequest(text="Hello.", speaker=self.settings.default_speaker)))
            log.info("Warmup complete.")
        except Exception:  # pragma: no cover - warmup is best-effort
            log.exception("Warmup failed (continuing anyway).")

    def _method_and_kwargs(self, req: TTSRequest):
        """Pick the qwen-tts method and kwargs for the request's mode."""
        mode = req.resolve_mode()
        lang = req.language or self.settings.default_language
        if mode == "voice_clone":
            return self.model.generate_voice_clone, dict(
                text=req.text, language=lang,
                ref_audio=req.ref_audio, ref_text=req.ref_text or "",
            )
        if mode == "voice_design":
            return self.model.generate_voice_design, dict(
                text=req.text, language=lang, instruct=req.instruct,
            )
        return self.model.generate_custom_voice, dict(
            text=req.text, language=lang,
            speaker=req.speaker or self.settings.default_speaker,
            instruct=req.instruct or "",
        )

    def _raw_stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        """Call the library. THE one adaptation point for the real API.

        Tries native streaming (``stream=True``); if the installed version
        doesn't support it, falls back to a full generation and re-chunks it so
        callers still get a stream.
        """
        method, kwargs = self._method_and_kwargs(req)
        try:
            result = method(stream=True, **kwargs)
        except TypeError:
            result = None

        if result is not None and hasattr(result, "__iter__") \
                and not isinstance(result, tuple):
            # Native streaming: a generator/iterator of chunks.
            for chunk in result:
                yield _as_float_mono(_chunk_audio(chunk))
            return

        # Fallback: generate the whole clip, then slice it into frames.
        wavs, sr = method(**kwargs)
        self.sample_rate = int(sr) or self.sample_rate
        audio = _as_float_mono(_first_wav(wavs))
        step = self.settings.stream_chunk_samples
        for start in range(0, len(audio), step):
            yield audio[start:start + step]

    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        for chunk in self._raw_stream(req):
            if chunk.size:
                yield chunk


# --------------------------------------------------------------------------- #
# Normalisation helpers (tolerant of the various shapes libraries return)
# --------------------------------------------------------------------------- #
def _first_wav(wavs) -> np.ndarray:
    arr = np.asarray(wavs)
    if arr.ndim >= 2:  # batch of clips -> take the first
        arr = arr[0]
    return arr


def _chunk_audio(chunk) -> np.ndarray:
    # A streaming chunk may be a bare array or a (array, sr) tuple.
    if isinstance(chunk, tuple):
        chunk = chunk[0]
    return np.asarray(chunk)


def _as_float_mono(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim > 1:  # (channels, n) or (n, channels) -> mono
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[-1] else 1)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
def build_engine(settings: Settings, model_id: str | None = None) -> TTSEngine:
    if settings.backend == "qwen":
        return QwenEngine(settings, model_id)
    return MockEngine(settings, model_id)
