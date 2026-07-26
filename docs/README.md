# Documentation

Reference documentation for the Qwen3-TTS streaming server.

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | Components, request lifecycle, concurrency & cancellation model, how to swap in the real model |
| [api.md](api.md) | HTTP REST endpoints (native + OpenAI-compatible), request/response schemas, examples |
| [websocket.md](websocket.md) | Bidirectional WebSocket protocol, message types, sequence diagrams, cancellation |
| [configuration.md](configuration.md) | Every environment variable, model allow-listing, auth |
| [deployment.md](deployment.md) | Running locally, the real GPU backend, Docker, scaling, troubleshooting |

## Quick map of the codebase

```
app/
  config.py     Settings (env-driven) + model allow-list
  schemas.py    Pydantic request models, speaker/language lists, validation
  audio.py      float->PCM16 conversion, WAV framing (streaming + exact-size)
  engine.py     TTSEngine ABC, MockEngine, QwenEngine (the one model-call site)
  streaming.py  Synthesizer: thread<->async bridge, serialization, cancellation
  manager.py    EngineManager: lazy per-model Synthesizer registry
  main.py       FastAPI app: REST, OpenAI, and WebSocket endpoints
```

## Terminology

- **Engine** — wraps a single loaded model and produces float audio chunks.
- **Synthesizer** — an engine plus the concurrency controls (queue, semaphore,
  cancellation) and the byte-level streaming logic around it.
- **Manager** — the registry that owns one Synthesizer per model id.
- **Chunk / frame** — one unit of streamed audio (`TTS_STREAM_CHUNK_SAMPLES`
  samples ≈ 50 ms at 24 kHz), sent as 16-bit little-endian PCM.
