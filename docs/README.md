# Documentation

Reference documentation for the Qwen3-TTS streaming server.

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | Components, request lifecycle, concurrency & cancellation model, how to swap in the real model |
| [api.md](api.md) | HTTP REST endpoints (native + OpenAI-compatible), request/response schemas, examples |
| [websocket.md](websocket.md) | Bidirectional WebSocket protocol, message types, sequence diagrams, cancellation |
| [configuration.md](configuration.md) | Every environment variable, model allow-listing, auth |
| [deployment.md](deployment.md) | Running locally, the real GPU backend, Docker, scaling, troubleshooting |

## Proposals

Design notes for changes that are not built yet. Everything above this line documents the
server as it is; everything below describes intent.

| Proposal | Status | Contents |
|---|---|---|
| [proposals/timing-marks.md](proposals/timing-marks.md) | implemented | Emit word/phoneme timings alongside streamed audio, for clients that animate in sync (lip-sync, captions) |
| [proposals/vad-auto-flush.md](proposals/vad-auto-flush.md) | implemented | Server-side speech detection on `/v1/sst_ws`, so a turn ends when the speaker stops instead of when the client says so (hands-free turn-taking, barge-in) |
| [proposals/concurrent-generation.md](proposals/concurrent-generation.md) | implemented | Run N instances of one model behind a pool instead of `Semaphore(1)`, to fill a GPU that a single small model leaves idle |

## Quick map of the codebase

```
app/
  config.py     Settings (env-driven): default backend, enabled backends, models
  schemas.py    Pydantic request models (model/speaker/language/speed/mode)
  audio.py      float->PCM16 conversion, WAV framing (streaming + exact-size)
  engine.py     Engine registry + TTSEngine ABC: Mock/Qwen/Kokoro/Dia engines
  vad.py        VAD registry + TurnDetector: speech detection and endpointing
  streaming.py  Synthesizer: thread<->async bridge, serialization, cancellation
  manager.py    EngineManager: multi-backend catalog + lazy per-model routing
  main.py       FastAPI app: REST, OpenAI, and WebSocket endpoints
```

## Terminology

- **Engine** — wraps a single loaded model and produces float audio chunks; each
  backend (Qwen, Kokoro, Dia, …) is a registered `TTSEngine` subclass.
- **Synthesizer** — an engine pool plus the concurrency controls (queue,
  cancellation) and the byte-level streaming logic around it.
- **Replica / pool** — one of N interchangeable instances of the same model
  (`TTS_ENGINE_REPLICAS`). A generation borrows one for its duration; a pool of
  one is the historical "serialise everything" behaviour.
- **Manager** — hosts multiple backends and routes each request's `model` to the
  right lazily-loaded Synthesizer (one per `backend:model_id`).
- **Chunk / frame** — one unit of streamed audio (`TTS_STREAM_CHUNK_SAMPLES`
  samples ≈ 50 ms at 24 kHz), sent as 16-bit little-endian PCM.
- **VAD** — a per-frame speech detector (`silero`, `energy`, `webrtc`). It only
  answers "is this frame speech?".
- **Turn** — one utterance on `/v1/sst_ws`, from speech onset to the trailing
  silence that ends it. `TurnDetector` is the hysteresis that turns flickering
  VAD output into turn boundaries; everything worth tuning lives there.
