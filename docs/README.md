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

| Proposal | Contents |
|---|---|
| [proposals/timing-marks.md](proposals/timing-marks.md) | Emit word/phoneme timings alongside streamed audio, for clients that animate in sync (lip-sync, captions) |

## Quick map of the codebase

```
app/
  config.py     Settings (env-driven): default backend, enabled backends, models
  schemas.py    Pydantic request models (model/speaker/language/speed/mode)
  audio.py      float->PCM16 conversion, WAV framing (streaming + exact-size)
  engine.py     Engine registry + TTSEngine ABC: Mock/Qwen/Kokoro/Dia engines
  streaming.py  Synthesizer: thread<->async bridge, serialization, cancellation
  manager.py    EngineManager: multi-backend catalog + lazy per-model routing
  main.py       FastAPI app: REST, OpenAI, and WebSocket endpoints
```

## Terminology

- **Engine** — wraps a single loaded model and produces float audio chunks; each
  backend (Qwen, Kokoro, Dia, …) is a registered `TTSEngine` subclass.
- **Synthesizer** — an engine plus the concurrency controls (queue, semaphore,
  cancellation) and the byte-level streaming logic around it.
- **Manager** — hosts multiple backends and routes each request's `model` to the
  right lazily-loaded Synthesizer (one per `backend:model_id`).
- **Chunk / frame** — one unit of streamed audio (`TTS_STREAM_CHUNK_SAMPLES`
  samples ≈ 50 ms at 24 kHz), sent as 16-bit little-endian PCM.
