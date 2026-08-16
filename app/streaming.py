"""Bridge the engine's blocking, GPU-bound generator to an async HTTP stream.

One model instance is not safe to run concurrently, so a generation must have
one to itself. Generation happens on a worker thread; produced audio chunks flow
through a thread-safe queue and are handed to the async response as they arrive,
so the client starts receiving audio before synthesis finishes.

How many can run at once is :class:`EnginePool`'s business: with the default one
replica it is exactly the old ``Semaphore(1)``, and with more it is that many
concurrent generations against that many independent model instances.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Iterator

from .audio import float_to_pcm16, wav_header
from .config import Settings
from .engine import Mark, TTSEngine
from .schemas import TTSRequest

log = logging.getLogger("tts.streaming")

_SENTINEL = object()


class EnginePool:
    """N interchangeable instances of one model; a generation borrows one.

    The free-queue *is* the semaphore: ``acquire()`` blocks when every replica
    is busy, exactly as ``Semaphore(1)`` blocked when the single engine was.
    A pool of one is therefore byte-for-byte the old behaviour, which is why
    ``TTS_ENGINE_REPLICAS`` defaults to 1 and costs nothing.

    Replicas past the first are built in the background by :meth:`fill`, so the
    first request does not pay for all of them. A partially-filled pool is just
    a smaller pool.
    """

    def __init__(
        self, primary: TTSEngine,
        build: Callable[[], TTSEngine] | None = None, size: int = 1,
    ):
        #: A representative instance. Owns the metadata (sample rate, model id)
        #: and is the one warmed up synchronously; replicas are identical.
        self.primary = primary
        self._build = build
        self._target = max(1, size)
        self._built = 1
        self._free: asyncio.Queue[TTSEngine] = asyncio.Queue()
        self._free.put_nowait(primary)

    @property
    def size(self) -> int:
        """Replicas actually built (may trail ``target`` while filling)."""
        return self._built

    @property
    def target(self) -> int:
        return self._target

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[TTSEngine]:
        engine = await self._free.get()
        try:
            yield engine
        finally:
            self._free.put_nowait(engine)

    async def fill(self) -> None:
        """Build the remaining replicas, one at a time, off the event loop."""
        loop = asyncio.get_running_loop()
        while self._built < self._target:
            try:
                engine = await loop.run_in_executor(None, self._build_one)
            except Exception:
                # A replica that won't build is a smaller pool, not an outage.
                log.exception(
                    "Replica build failed; continuing with %d instance(s).",
                    self._built,
                )
                self._target = self._built
                return
            self._built += 1
            self._free.put_nowait(engine)
            log.info("Replica %d/%d ready for %s",
                     self._built, self._target, self.primary.model_id)

    def _build_one(self) -> TTSEngine:
        assert self._build is not None, "pool has no builder"
        engine = self._build()
        engine.warmup()   # a cold replica would make every Nth request slow
        return engine


class Synthesizer:
    """Owns the engine pool plus the concurrency controls around it."""

    def __init__(self, engine: TTSEngine | EnginePool, settings: Settings):
        # Accepts a bare engine (the common case, and what tests construct) or
        # a pre-built pool from the manager.
        self.pool = engine if isinstance(engine, EnginePool) else EnginePool(engine)
        #: Representative instance — metadata and capability flags read off it.
        self.engine = self.pool.primary
        self.settings = settings
        self._inflight = 0                        # queued + running requests

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    @property
    def model_id(self) -> str:
        return self.engine.model_id

    @property
    def at_capacity(self) -> bool:
        """True if the generation queue is full (used for admission control).

        ``max_queue`` bounds requests *in flight* — queued plus running — so
        with N replicas N of them are generating and the rest are waiting.
        Deliberately absolute: the depth a caller will tolerate is a property
        of the caller, not of how many replicas happen to be loaded.
        """
        return self._inflight >= self.settings.max_queue

    def _run_blocking(
        self, engine: TTSEngine, req: TTSRequest, q: "queue.Queue",
        cancel: "threading.Event | None", *, marked: bool = False,
    ) -> None:
        """Worker-thread body: push PCM byte chunks onto ``q``.

        Takes the engine as an argument rather than reading ``self.engine``:
        with a pool, the instance this generation owns is the one it borrowed.

        With ``marked=True`` the engine's marked stream is used instead and
        the items are ``(pcm_bytes, marks)`` pairs.
        """
        try:
            if marked:
                for chunk, marks in engine.stream_marked(req):
                    if cancel is not None and cancel.is_set():
                        break
                    q.put((float_to_pcm16(chunk), marks))
            else:
                for chunk in engine.stream(req):
                    if cancel is not None and cancel.is_set():
                        break
                    q.put(float_to_pcm16(chunk))
        except Exception as exc:  # surface errors to the consumer
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    async def _pcm_chunks(
        self, req: TTSRequest, cancel: "threading.Event | None" = None,
        *, marked: bool = False,
    ) -> AsyncIterator:
        """Yield PCM byte frames as the worker produces them.

        If ``cancel`` is set mid-stream, the worker stops at the next chunk
        boundary and the queue is drained so the thread exits cleanly.
        """
        if self._inflight >= self.settings.max_queue:
            raise RuntimeError("server busy: generation queue is full")

        self._inflight += 1
        try:
            async with self.pool.acquire() as engine:
                loop = asyncio.get_running_loop()
                q: "queue.Queue" = queue.Queue(maxsize=8)  # bounded => backpressure
                worker = threading.Thread(
                    target=self._run_blocking, args=(engine, req, q, cancel),
                    kwargs={"marked": marked}, daemon=True,
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

    async def stream_marked_pcm(
        self, req: TTSRequest, cancel: "threading.Event | None" = None,
    ) -> AsyncIterator[tuple[bytes, list[Mark]]]:
        """Yield ``(PCM frame, marks)`` pairs: the frame plus any timing marks
        whose range falls inside it.

        Used by the WebSocket handler, which has a JSON side channel for the
        marks that HTTP chunked responses lack. For engines without timing
        support the marks list is always empty.
        """
        async for item in self._pcm_chunks(req, cancel, marked=True):
            yield item

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

    def __init__(self, engine: "SSEngine", settings):  # accepts _STTSettings
        self.engine = engine            # the loaded SSEngine instance
        self._language: str | None = None  # per-call language override set by handler
        self.settings = settings        # shared server config (_STTSettings)
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
