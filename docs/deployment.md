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
- Generation is **serialised per model per process** (a model isn't
  concurrency-safe). To serve more throughput, run **multiple replicas**, each
  with its own GPU, behind a load balancer.
- Do **not** naively bump `uvicorn --workers` on a single GPU expecting linear
  speedup — each worker loads its own copy of the model into GPU memory.
- Tune `TTS_MAX_QUEUE` to bound queueing latency; excess requests get `503` so
  clients can retry/backoff rather than wait unboundedly.

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
| WS closes immediately (1008) | Auth enabled but key missing/wrong; pass `?api_key=` or the bearer header. |
