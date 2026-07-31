"""Bridge the engine's blocking, GPU-bound generator to an async HTTP stream.

The model is not safe to run concurrently, so a semaphore serialises access.
Generation happens on a worker thread; produced audio chunks flow through a
thread-safe queue and are handed to the async response as they arrive, so the
client starts receiving audio before synthesis finishes.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import AsyncIterator, Iterator

from .audio import float_to_pcm16, wav_header
from .config import Settings
from .engine import TTSEngine
from .schemas import TTSRequest

log = logging.getLogger("tts.streaming")

_SENTINEL = object()


class Synthesizer:
    """Owns the engine plus the concurrency controls around it."""

    def __init__(self, engine: TTSEngine, settings: Settings):
        self.engine = engine
        self.settings = settings
        self._gpu_lock = asyncio.Semaphore(1)   # one generation at a time
        self._inflight = 0                        # queued + running requests

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    @property
    def model_id(self) -> str:
        return self.engine.model_id

    @property
    def at_capacity(self) -> bool:
        """True if the generation queue is full (used for admission control)."""
        return self._inflight >= self.settings.max_queue

    def _run_blocking(
        self, req: TTSRequest, q: "queue.Queue",
        cancel: "threading.Event | None",
    ) -> None:
        """Worker-thread body: push PCM byte chunks onto ``q``."""
        try:
            for chunk in self.engine.stream(req):
                if cancel is not None and cancel.is_set():
                    break
                q.put(float_to_pcm16(chunk))
        except Exception as exc:  # surface errors to the consumer
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    async def _pcm_chunks(
        self, req: TTSRequest, cancel: "threading.Event | None" = None
    ) -> AsyncIterator[bytes]:
        """Yield PCM byte frames as the worker produces them.

        If ``cancel`` is set mid-stream, the worker stops at the next chunk
        boundary and the queue is drained so the thread exits cleanly.
        """
        if self._inflight >= self.settings.max_queue:
            raise RuntimeError("server busy: generation queue is full")

        self._inflight += 1
        try:
            async with self._gpu_lock:
                loop = asyncio.get_running_loop()
                q: "queue.Queue" = queue.Queue(maxsize=8)  # bounded => backpressure
                worker = threading.Thread(
                    target=self._run_blocking, args=(req, q, cancel), daemon=True
                )
                worker.start()
                stopped = False
                while True:
                    item = await loop.run_in_executor(None, q.get)
                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    if stopped:
                        continue  # draining after cancel; don't emit
                    if cancel is not None and cancel.is_set():
                        stopped = True  # keep draining until the worker's sentinel
                        continue
                    yield item
        finally:
            self._inflight -= 1

    async def stream_response(
        self, req: TTSRequest, response_format: str,
        cancel: "threading.Event | None" = None,
    ) -> AsyncIterator[bytes]:
        """Yield an HTTP body: WAV (header first) or raw PCM."""
        if response_format == "wav":
            yield wav_header(self.sample_rate)  # open-ended streaming header
        async for pcm in self._pcm_chunks(req, cancel):
            yield pcm

    async def synthesize_bytes(self, req: TTSRequest) -> bytes:
        """Collect the full PCM buffer (used for non-streaming responses)."""
        parts = [pcm async for pcm in self._pcm_chunks(req)]
        return b"".join(parts)


# --------------------------------------------------------------------------- #
# Transcriber — SSEngine bridge (audio-in → text-out)                         #
# --------------------------------------------------------------------------- #

_SENTINEL_TXT = object()


class Transcriber:
    """Wraps an :class:`~app.engine.SSEngine` with async concurrency controls.

    Mirrors :class:`Synthesizer` but for **speech-to-text**: consumes audio bytes,
    yields text segments as JSON-serializable strings, and supports cancellation
    between segments (like TTS does for PCM frames).
    """

    def __init__(self, engine: "SSEngine", settings):  # accepts _SSTSettings
        self.engine = engine            # the loaded SSEngine instance
        self._language: str | None = None  # per-call language override set by handler
        self.settings = settings        # shared server config (_SSTSettings)
        self._gpu_lock = asyncio.Semaphore(1)   # one generation at a time
        self._inflight = 0                        # queued + running requests

    @property
    def sample_rate(self) -> int:
        """Input audio sample-rate hint."""
        return self.engine.SAMPLE_RATE or getattr(self.settings, "sample_rate", 16000)

    @property
    def model_id(self) -> str:
        return self.engine.model_id

    @property
    def at_capacity(self) -> bool:
        """True if the transcription queue is full (admission control)."""
        return self._inflight >= getattr(self.settings, "max_queue", 32)

    # -- language override (set by HTTP handler per-call) ----------------------

    def set_language(self, lang: str | None) -> None:
        """Override the language for the next transcription call."""
        self._language = lang

    # -- worker thread --------------------------------------------------------

    def _run_blocking(
        self, audio_bytes: bytes, q: "queue.Queue", cancel: "threading.Event | None",
        lang: str | None = None,
    ) -> None:
        """Worker-thread body: produce text segments into the queue."""
        try:
            effective_lang = lang if lang is not None else getattr(self.settings, "default_language", None)
            for segment in self.engine.stream_transcribe(audio_bytes, language=effective_lang):
                if cancel is not None and cancel.is_set():
                    break
                q.put(segment)
        except Exception as exc:  # surface errors to the consumer
            q.put(exc)
        finally:
            q.put(_SENTINEL_TXT)

    # -- async consumer -------------------------------------------------------

    async def _text_segments(
        self, audio_bytes: bytes, cancel: "threading.Event | None" = None,
        lang: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text segments as the worker produces them."""
        if self._inflight >= getattr(self.settings, "max_queue", 32):
            raise RuntimeError("server busy: transcription queue is full")

        self._inflight += 1
        try:
            async with self._gpu_lock:
                loop = asyncio.get_running_loop()
                q: "queue.Queue" = queue.Queue(maxsize=8)
                effective_lang = lang if lang is not None else self._language
                worker = threading.Thread(
                    target=self._run_blocking, args=(audio_bytes, q, cancel), kwargs={"lang": effective_lang}, daemon=True)
                worker.start()
                stopped = False
                while True:
                    item = await loop.run_in_executor(None, q.get)
                    if item is _SENTINEL_TXT:
                        break
                    if isinstance(item, Exception):
                        raise item
                    if stopped:
                        continue
                    if cancel is not None and cancel.is_set():
                        stopped = True
                        continue
                    yield item
        finally:
            self._inflight -= 1

    # -- streaming bridge (HTTP) -----------------------------------------------

    async def stream_text(
        self, audio_bytes: bytes,
        cancel: "threading.Event | None" = None,
        language: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield text segments for HTTP streaming."""
        effective_lang = language if language is not None else self._language
        async for seg in self._text_segments(audio_bytes, cancel, lang=effective_lang):
            yield seg

    # -- non-streaming ---------------------------------------------------------

    async def transcribe(
        self, audio_bytes: bytes,
    ) -> str:
        """Consume the full segment stream and return one concatenated transcript."""
        parts = [seg async for seg in self._text_segments(audio_bytes)]
        return " ".join(parts)
