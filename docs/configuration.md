# Configuration

All settings are environment variables prefixed with `TTS_`, or entries in a
local `.env` file (see [`.env.example`](../.env.example)). Loaded once at
startup via `app/config.py`.

## Reference

### Backend
| Variable | Default | Description |
|---|---|---|
| `TTS_BACKEND` | `mock` | **Default** engine: `mock`, `qwen` (Qwen3-TTS, GPU), `kokoro` (Kokoro-82M, CPU/GPU), `dia` (Dia-1.6B, GPU). Validated at startup. |
| `TTS_BACKENDS` | *(empty)* | Comma-separated extra backends clients may select **by name** (default backend always enabled). |

### Model loading (real backends)
| Variable | Default | Description |
|---|---|---|
| `TTS_MODEL_ID` | *(empty)* | Default model for the default backend. Empty = that backend's own default (e.g. `hexgrad/Kokoro-82M`). |
| `TTS_MODELS` | *(empty)* | Comma-separated **allow-list** of extra selectable models, each `backend:model_id`. |
| `TTS_DEVICE` | `cuda:0` | Torch device (`device_map`); GPU backends. |
| `TTS_DTYPE` | `bfloat16` | `bfloat16` \| `float16` \| `float32` (Qwen). |
| `TTS_ATTN_IMPLEMENTATION` | `sdpa` | Qwen attention backend. `sdpa` needs no build; `flash_attention_2` is an optional speedup. |

### Audio
| Variable | Default | Description |
|---|---|---|
| `TTS_SAMPLE_RATE` | `24000` | Output sample rate in Hz. |
| `TTS_STREAM_CHUNK_SAMPLES` | `1200` | Samples per streamed frame (~50 ms @ 24 kHz). Lower = lower latency, more overhead. |

### Generation defaults
| Variable | Default | Description |
|---|---|---|
| `TTS_DEFAULT_LANGUAGE` | `Auto` | Used when a request omits `language`. |
| `TTS_DEFAULT_SPEAKER` | `Vivian` | Used when a request omits `speaker`. |

### Concurrency
| Variable | Default | Description |
|---|---|---|
| `TTS_MAX_QUEUE` | `32` | Max in-flight requests (per model) before returning `503`. |

### Server
| Variable | Default | Description |
|---|---|---|
| `TTS_HOST` | `0.0.0.0` | Bind host (when run via `python -m app.main`). |
| `TTS_PORT` | `8000` | Bind port. |
| `TTS_API_KEYS` | *(empty)* | Comma-separated bearer tokens. Empty = auth disabled. |

## Backends & model selection

A single server can host several backends; the **client** picks one per request.

- **Default backend/model:** `TTS_BACKEND` + `TTS_MODEL_ID` (empty `TTS_MODEL_ID`
  → that backend's own default, e.g. `kokoro` → `hexgrad/Kokoro-82M`). Used when
  a request omits `model`.
- **Enabling more backends for clients:** `TTS_BACKENDS=kokoro,dia` makes those
  selectable by name. The default backend is always enabled.
- **Adding specific models:** `TTS_MODELS`, comma-separated, each
  `backend:model_id` (a bare `model_id` uses the default backend; a bare
  `backend` means that backend's default model).

A request's `model` field selects by:

1. **backend name** (`"kokoro"`, `"dia"`, `"qwen"`) → that backend's default model;
2. **model id** (`"hexgrad/Kokoro-82M"`) present in the catalog;
3. **`backend:model_id`** for an enabled backend + catalogued id;
4. omitted / generic alias (`default`, `auto`, `tts-1`, …) → the default model.

Anything else returns `400`, so a request can't trigger an arbitrary
multi-gigabyte download. Each selected model is **loaded lazily** on first use
and cached (mind GPU memory when enabling several large models). `GET /v1/models`
lists what's selectable; `GET /v1/voices?model=…` shows a specific model's voices.

With `TTS_BACKEND=mock`, any model name is accepted (nothing is downloaded) and
echoed in the `X-Model` header / WS `start` message — handy for testing
client-side model-selection logic.

## Authentication

Set `TTS_API_KEYS=key1,key2`. Then:

- REST / OpenAI / WS (header): `Authorization: Bearer key1`.
- WebSocket from a browser (no custom headers): `...?api_key=key1`.

Empty `TTS_API_KEYS` disables auth entirely (fine for local/dev; put a gateway
in front for anything public).

## Example `.env`

```dotenv
# Default to Qwen, but also let clients pick Kokoro or Dia per request.
TTS_BACKEND=qwen
TTS_BACKENDS=kokoro,dia
TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
# optional extra specific models:
# TTS_MODELS=kokoro:hexgrad/Kokoro-82M
TTS_DEVICE=cuda:0
TTS_DTYPE=bfloat16
TTS_STREAM_CHUNK_SAMPLES=1200
TTS_MAX_QUEUE=16
TTS_API_KEYS=super-secret-key
```
