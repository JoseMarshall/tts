# Architecture

## Overview

The server is a thin, backend-agnostic HTTP/WebSocket layer over one or more
loaded TTS models. Everything is organised so that **the only code that talks to
the model library lives in one method**, and all framing/streaming/transport is
independent of which model (or even which backend) is in use.

```
                         ┌────────────────────────────────────────────┐
  HTTP / WebSocket  ───▶ │  main.py  (FastAPI routes)                  │
                         │    ├─ REST: /v1/tts, /v1/tts/stream         │
                         │    ├─ OpenAI: /v1/audio/speech              │
                         │    └─ WebSocket: /v1/tts/ws                 │
                         └───────────────┬────────────────────────────┘
                                         │ resolve model
                         ┌───────────────▼────────────────────────────┐
                         │  EngineManager (manager.py)                 │
                         │    model id  ->  Synthesizer  (lazy build)  │
                         └───────────────┬────────────────────────────┘
                                         │
                         ┌───────────────▼────────────────────────────┐
                         │  Synthesizer (streaming.py)                 │
                         │    • semaphore(1): serialise model access   │
                         │    • worker thread -> bounded queue -> async│
                         │    • cancellation between chunks            │
                         └───────────────┬────────────────────────────┘
                                         │ engine.stream(req)
                         ┌───────────────▼────────────────────────────┐
                         │  TTSEngine (engine.py)                      │
                         │    MockEngine   |   QwenEngine._raw_stream  │
                         └─────────────────────────────────────────────┘
```

## Components

### `TTSEngine` (`engine.py`)
Abstract base. The single primitive an engine implements is
`stream(req) -> Iterator[np.ndarray]`, yielding float32 mono chunks in `[-1, 1]`
at `sample_rate`. Non-streaming synthesis is just "consume and concatenate".

Engines self-register via the `@register` decorator into a name→class registry;
`build_engine()` looks up `TTS_BACKEND` there, and `GET /v1/voices` reads each
class's `capabilities()` (speakers, languages, defaults). Class metadata
(`NAME`, `SAMPLE_RATE`, `SPEAKERS`, `LANGUAGES`, `DEFAULT_SPEAKER`,
`DEFAULT_LANGUAGE`) keeps everything else backend-agnostic — see
[Adding a backend](#adding-a-backend).

- **`MockEngine`** — a dependency-free chord generator whose length scales with
  text length. Lets the whole stack run and be tested without a GPU or model.
- **`QwenEngine`** — loads a `qwen-tts` model. The **one adaptation point** is
  `QwenEngine._raw_stream`: it tries native `stream=True` generation and falls
  back to full generation re-chunked into frames if the installed library
  doesn't support streaming. Normalisation helpers tolerate the various shapes
  (batched arrays, `(array, sr)` tuples, torch tensors, int vs float).
- **`KokoroEngine`** — loads Kokoro-82M via `KPipeline` (one pipeline cached per
  language code). Kokoro yields per text segment, which the engine re-slices
  into fixed frames. Preset voices only; cloning/design raise `ValueError`→`400`.
- **`DiaEngine`** — loads Dia2-1B (`dia2` package, `Dia2.from_repo`). Dialogue
  via `[S1]`/`[S2]` tags, no preset speakers; voice cloning by prefix
  conditioning on reference audio. Outputs Mimi's 24 kHz and hands back one
  finished waveform, so it generates then re-chunks; the result's word
  timestamps become timing marks (`SUPPORTS_MARKS`).

### `Synthesizer` (`streaming.py`)
Owns an engine pool plus the concurrency machinery:

- An **`EnginePool`** holds N interchangeable instances of the model; a
  generation borrows one for its duration. The free-queue *is* the semaphore —
  `acquire()` blocks when every replica is busy — so the default
  `TTS_ENGINE_REPLICAS=1` is exactly the old `Semaphore(1)`, and one model
  instance still never runs two generations at once.
- Replicas past the first are built **in the background** (`pool.fill()`), so
  the first request doesn't pay for all of them; a partially-filled pool is just
  a smaller pool. Each replica is warmed, so no request lands on a cold one.
  `>1` requires the backend to declare `SUPPORTS_REPLICAS`, because two Python
  objects sharing a global (espeak-ng, a module cache) are not two independent
  models — and that failure is wrong output, not a crash.
- Generation is blocking and GPU-bound, so it runs on a **worker thread**.
  Produced PCM byte frames go through a **bounded `queue.Queue`** (backpressure —
  the worker blocks if the client is slow), and an async consumer pulls from it
  via `run_in_executor`, yielding frames as they arrive.
- A `threading.Event` provides **cancellation**: when set, the worker stops at
  the next chunk boundary and the consumer drains the queue to the sentinel so
  the thread exits cleanly (no leak).
- `_inflight` bounds how many requests may queue before we reject with `503`
  (`TTS_MAX_QUEUE`).

### `EngineManager` (`manager.py`)
Hosts **multiple backends at once** and routes each request to the right one.

- A **catalog** of `ModelSpec(backend, model_id)` is built at startup from the
  default (`TTS_BACKEND`/`TTS_MODEL_ID`), `TTS_BACKENDS` (extra client-selectable
  backends), and `TTS_MODELS` (`backend:model_id` entries).
- `resolve(model)` maps a request's `model` to a `ModelSpec`: a **backend name**
  → that backend's default model; a catalogued **model id**; `backend:model_id`;
  or a generic alias → the default. Anything else raises `UnknownModelError`
  (→`400`) so a request can't trigger an arbitrary multi-gigabyte download. The
  `mock` default backend is permissive (fabricates any name) for tests/dev.
- Each `ModelSpec` gets its own `Synthesizer`, built **lazily** on first use in a
  thread executor under an `asyncio.Lock` (concurrent first-requests load once)
  and cached by `backend:model_id`. Different models — even different backends —
  each have their own engine and semaphore, so they can run concurrently (mind
  GPU memory). `/v1/voices` reads a spec's backend `capabilities()` without
  loading the model.

### Routes (`main.py`)
Stateless glue: parse/validate the request, resolve the `Synthesizer` for the
requested model, and stream its output. The WebSocket handler additionally runs
a receive loop and a synthesis task concurrently so it can accept `cancel`
frames mid-utterance.

## Request lifecycle (streaming HTTP)

1. Client `POST /v1/tts/stream` with `text`, `model`, `language`, voice params.
2. Pydantic validates the body (including the `language` allow-list).
3. `EngineManager.get(model)` returns (building if needed) the `Synthesizer`.
4. `Synthesizer.stream_response` emits a WAV header (for `wav`) then acquires the
   semaphore, starts the worker thread, and yields PCM frames as produced.
5. FastAPI's `StreamingResponse` writes each frame to the socket immediately, so
   the client hears audio before synthesis finishes.

## Concurrency & cancellation model

- **Per model, one generation at a time** (semaphore). Different models can run
  concurrently (each has its own `Synthesizer`); mind GPU memory if you enable
  several large models at once.
- **Backpressure**: the bounded queue means a slow client throttles generation
  rather than letting audio pile up in memory.
- **Cancellation** (WebSocket `cancel`, or client disconnect): cooperative at
  chunk boundaries. With a fast mock you may still receive a burst already in
  flight; with a real, slower model, generation stops promptly.
- **Overload**: past `TTS_MAX_QUEUE` in-flight requests, new ones get `503`.

## Audio format

- Internal: float32 mono, `[-1, 1]`, at `TTS_SAMPLE_RATE` (24000 by default).
- On the wire: 16-bit little-endian PCM. `wav` wraps it in a WAV container;
  `pcm` sends it raw with the rate in the `X-Sample-Rate` header.
- **Streaming WAV** uses open-ended (`0xFFFFFFFF`) size fields since the length
  isn't known up front. Most players accept this. The non-streaming `/v1/tts`
  writes exact sizes. See [api.md](api.md#audio-formats) for the trade-off.

## Adding a backend

Everything except the engine class is model-agnostic, so a new backend is one
registered `TTSEngine` subclass in `app/engine.py`:

```python
@register
class MyEngine(TTSEngine):
    NAME = "mybackend"          # the TTS_BACKEND value that selects it
    SAMPLE_RATE = 24000
    SPEAKERS = [...]            # advertised by GET /v1/voices
    LANGUAGES = [...]
    DEFAULT_SPEAKER = "..."
    DEFAULT_LANGUAGE = "..."

    def __init__(self, settings, model_id=None):
        super().__init__(settings, model_id)
        ...                     # load the model; import heavy deps *here*

    def stream(self, req):
        for chunk in my_model.generate(req.text, voice=self._speaker(req)):
            yield chunk         # float32 mono numpy at SAMPLE_RATE
```

Guidelines:
- Keep heavy imports (`torch`, the model package) **inside `__init__`** so the
  server and tests run without them when another backend is selected.
- Use the `self._speaker(req)` / `self._language(req)` helpers — they resolve
  request value → `TTS_DEFAULT_*` → the engine's own default.
- Raise `ValueError` for anything the model can't do (unsupported mode, unknown
  voice/language); the API maps it to `400`.
- Normalise outputs through `_as_float_mono` (handles torch tensors, int PCM,
  channel layouts). Emit chunks around `settings.stream_chunk_samples` for
  smooth streaming; if the model isn't natively streaming, generate fully and
  slice (see `QwenEngine._raw_stream`'s fallback).

That's it — routing, WAV/PCM framing, the WebSocket protocol, cancellation, the
allow-list and lazy multi-model loading all work unchanged. Select it with
`TTS_BACKEND=mybackend`.

### The Qwen adaptation point
For Qwen specifically, only `QwenEngine._raw_stream` / `_method_and_kwargs` touch
the `qwen-tts` library; if a future version changes its streaming signature,
adapt those and nothing else.
