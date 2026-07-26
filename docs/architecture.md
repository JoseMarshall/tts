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

- **`MockEngine`** — a dependency-free chord generator whose length scales with
  text length. Lets the whole stack run and be tested without a GPU or model.
- **`QwenEngine`** — loads a `qwen-tts` model. The **one adaptation point** is
  `QwenEngine._raw_stream`: it tries native `stream=True` generation and falls
  back to full generation re-chunked into frames if the installed library
  doesn't support streaming. Normalisation helpers tolerate the various shapes
  (batched arrays, `(array, sr)` tuples, int vs float) a library might return.

### `Synthesizer` (`streaming.py`)
Owns one engine plus the concurrency machinery:

- A `Semaphore(1)` serialises generation, because a single model instance is not
  safe to run concurrently.
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
A registry mapping model id → `Synthesizer`, built lazily on first use. Building
is blocking (loading a real model onto the GPU), so it runs in a thread executor
under an `asyncio.Lock` so concurrent first-requests for the same model load it
once. Requested names are resolved through an alias table (generic names like
`qwen3-tts`, `tts-1`, `default` → the configured default) and, for the real
backend, validated against the allow-list so a request can't trigger an
arbitrary multi-gigabyte download.

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

## Swapping in the real model

Set `TTS_BACKEND=qwen` and install `torch qwen-tts soundfile` on a CUDA box.
Only `QwenEngine._raw_stream` (and `_method_and_kwargs`) touch the library — if
a future `qwen-tts` version changes its streaming signature, adapt those and
nothing else. The fallback path keeps the HTTP/WS API identical even if native
streaming is unavailable in the installed version.
