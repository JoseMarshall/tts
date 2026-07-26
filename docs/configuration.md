# Configuration

All settings are environment variables prefixed with `TTS_`, or entries in a
local `.env` file (see [`.env.example`](../.env.example)). Loaded once at
startup via `app/config.py`.

## Reference

### Backend
| Variable | Default | Description |
|---|---|---|
| `TTS_BACKEND` | `mock` | Registered engine name: `mock` (tone generator, no deps), `qwen` (Qwen3-TTS, GPU), `kokoro` (Kokoro-82M, CPU/GPU). Validated at startup. |

### Model loading (real backends)
| Variable | Default | Description |
|---|---|---|
| `TTS_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Default model, preloaded at startup. For Kokoro use `hexgrad/Kokoro-82M`. |
| `TTS_MODELS` | *(empty)* | Comma-separated **allow-list** of additional models a request may select. The default is always allowed. |
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

## Model selection & allow-listing

- A request selects a model with the `model` field (REST/OpenAI body, or WS
  `config`/`synthesize`). Omitting it uses `TTS_MODEL_ID`.
- Generic aliases — `qwen3-tts`, `tts-1`, `tts-1-hd`, `default`, `auto`, and the
  empty string — resolve to the default model. This keeps OpenAI clients working
  unchanged.
- With `TTS_BACKEND=qwen`, any non-alias model must appear in `model_list`
  (default + `TTS_MODELS`), otherwise the request gets `400`. This prevents a
  request from triggering an arbitrary multi-gigabyte download.
- Extra models are **loaded lazily** on first use and then cached. Each loaded
  model consumes GPU memory — only allow-list what fits.
- With `TTS_BACKEND=mock`, any model name is accepted (nothing is downloaded);
  the chosen name is echoed in the `X-Model` header / `start` message, which is
  handy for testing client model-selection logic.

## Authentication

Set `TTS_API_KEYS=key1,key2`. Then:

- REST / OpenAI / WS (header): `Authorization: Bearer key1`.
- WebSocket from a browser (no custom headers): `...?api_key=key1`.

Empty `TTS_API_KEYS` disables auth entirely (fine for local/dev; put a gateway
in front for anything public).

## Example `.env`

```dotenv
TTS_BACKEND=qwen
TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
TTS_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
TTS_DEVICE=cuda:0
TTS_DTYPE=bfloat16
TTS_STREAM_CHUNK_SAMPLES=1200
TTS_MAX_QUEUE=16
TTS_API_KEYS=super-secret-key
```
