# Deployment

## Local (mock backend, no GPU)

```bash
python -m venv .venv
# Windows:        . .venv/Scripts/activate
# Linux/macOS:    source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Verify:

```bash
bash client_examples/curl_examples.sh
python client_examples/stream_client.py "Hello streaming" out.wav
python client_examples/ws_client.py       # bidirectional WebSocket demo
```

## Real model (GPU)

On a machine with a CUDA GPU:

```bash
pip install -r requirements.txt
pip install -U torch qwen-tts soundfile   # torch build must match your CUDA

export TTS_BACKEND=qwen
export TTS_DEVICE=cuda:0
# optional: allow selecting more than the default model
export TTS_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The default model is loaded and warmed up at startup; extra allow-listed models
load lazily on first request. If `flash_attention_2` isn't available, set
`TTS_ATTN_IMPLEMENTATION=sdpa`.

> The only code touching the `qwen-tts` library is `QwenEngine._raw_stream` /
> `_method_and_kwargs` in `app/engine.py`. If a library update changes the
> streaming signature, adapt those; the fallback path already keeps the API
> working even without native streaming.

## Kokoro-82M (`kokoro`) — CPU or GPU

```bash
pip install -r requirements.txt
pip install -U kokoro soundfile
# espeak-ng system package (grapheme->phoneme, esp. non-English):
#   Windows: winget install eSpeak-NG.eSpeak-NG
#   Debian/Ubuntu: sudo apt-get install espeak-ng
#   macOS: brew install espeak-ng

export TTS_BACKEND=kokoro
export TTS_MODEL_ID=hexgrad/Kokoro-82M
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Kokoro is small (~350 MB) and runs on CPU, though a GPU is faster. It offers
preset voices only (no cloning/design). Voices/languages: `GET /v1/voices`.

## Docker

```bash
docker build -t qwen3-tts-server .
docker run --gpus all -p 8000:8000 \
  -e TTS_BACKEND=qwen -e TTS_DEVICE=cuda:0 \
  qwen3-tts-server
```

The provided `Dockerfile` uses a CUDA runtime base and installs
`torch qwen-tts soundfile`. For a CPU/mock image, any `python:3.11-slim` base
with just `requirements.txt` works.

## Production notes

### Workers & scaling

Two different levers for two different problems.

**Filling a GPU you already paid for → `TTS_ENGINE_REPLICAS`.** One model
instance is not concurrency-safe, so generation is serialised *per instance*.
A small model (Kokoro-82M is ~330 MB) leaves most of the card idle between short
forward passes, and the fix is more instances of it in the same process:

```
TTS_ENGINE_REPLICAS=4     # 4 concurrent generations, 4x the weights in VRAM
```

- VRAM scales linearly. Check `GET /health` → `replicas` for what actually
  loaded, and size it against `nvidia-smi`.
- Only backends declaring `SUPPORTS_REPLICAS` honour it; others log a warning
  and run one instance.
- Measure before committing: if throughput is flat from 1 → 2 → 4, the ceiling
  is somewhere other than serialisation.

**Failover, rolling restarts, more than one machine → separate processes.** Run
multiple replicas of the *server*, each with its own GPU, behind a load
balancer. Replicas inside one process give you none of that.

- Do **not** naively bump `uvicorn --workers` on a single GPU expecting linear
  speedup — each worker loads its own copy of every model. `TTS_ENGINE_REPLICAS`
  is the granular version of the same idea: it scales one model rather than the
  whole catalog.
- Tune `TTS_MAX_QUEUE` to bound queueing latency; excess requests get `503` so
  clients can retry/backoff rather than wait unboundedly. It counts requests
  in flight (queued plus running) and does not scale with replicas.

### Latency
- Lower `TTS_STREAM_CHUNK_SAMPLES` for snappier first-audio at the cost of more
  per-frame overhead; raise it to reduce overhead.
- Prefer `response_format:"pcm"` for real-time playback pipelines.

### WebSockets behind a proxy
- Ensure the proxy allows WebSocket upgrades and **disables response buffering**
  on the streaming/WS routes (e.g. nginx `proxy_buffering off;` and the
  `Upgrade`/`Connection` headers). Buffering defeats low-latency streaming.
- Set generous read timeouts for long-lived WS connections.

### Security
- Enable auth with `TTS_API_KEYS`, and/or terminate TLS and enforce auth at a
  gateway. The app itself does not do TLS.
- Voice cloning accepts `ref_audio` as a URL — if you expose cloning to
  untrusted users, restrict/validate those URLs to avoid SSRF.

## Health checks & observability

- Liveness/readiness: `GET /health` (includes backend, allow-listed models, and
  which models are currently loaded).
- Logs go to stdout at INFO (`app/main.py` configures logging). Model load,
  warmup, and per-model readiness are logged.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `400 Unknown model` | Model not in `TTS_MODELS` (real backend). Add it or use the default. |
| `422` on request | Missing `text`, or unsupported `language` (see `GET /v1/voices`). |
| `503 server busy` | Past `TTS_MAX_QUEUE` in-flight; scale out or raise the limit. |
| WAV shows wrong/infinite duration | Expected for streamed WAV (open-ended header). Use `/v1/tts` for exact sizes, or `pcm`. |
| Import error for `qwen_tts`/`torch` | Only needed with `TTS_BACKEND=qwen`; install them, or use `mock`. |
| flash-attn error at load | Set `TTS_ATTN_IMPLEMENTATION=sdpa`. |
| `Unknown TTS_BACKEND '…'` at startup | Must be a registered engine (`mock`/`qwen`/`kokoro`). Fix the typo. |
| Kokoro: espeak error / garbled non-English | Install `espeak-ng`; on Windows set `PHONEMIZER_ESPEAK_LIBRARY` to `libespeak-ng.dll`. |
| Kokoro: `400` on request | Kokoro is preset-voice only; `voice_clone`/`voice_design` aren't supported. |
| WS closes immediately (1008) | Auth enabled but key missing/wrong; pass `?api_key=` or the bearer header. |
