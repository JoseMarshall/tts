# Qwen3-TTS Streaming Server

A FastAPI server that exposes [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
models over HTTP with **low-latency chunked streaming**. Audio starts flowing to
the client as soon as the model emits its first packet, rather than after the
whole clip is generated.

It supports the model's three voice modes and ships with a dependency-free
**mock backend** so you can run, test and integrate against the API without a
GPU or the multi-gigabyte model.

## Features

- `POST /v1/tts/stream` — chunked streaming synthesis (WAV or raw PCM)
- `POST /v1/tts` — full-file (non-streaming) synthesis
- `POST /v1/audio/speech` — **OpenAI-compatible** endpoint (works with the OpenAI SDK)
- `WS /v1/tts/ws` — **bidirectional WebSocket streaming** (send text/control,
  receive audio concurrently, cancel mid-utterance)
- **Per-request model and language selection** (`model` + `language` fields);
  models are lazily loaded and restricted to an allow-list
- Three voice modes:
  - **Custom voice** (default) — a preset `speaker`, with `instruct` as an
    optional style modifier
  - **Voice cloning** — a `ref_audio` reference (path / URL / base64) + `ref_text`
  - **Voice design** — a voice from an `instruct` description alone; requires the
    dedicated VoiceDesign checkpoint and `mode: "voice_design"`

  Mode is inferred (`ref_audio` → cloning, else custom voice) unless you set
  `mode` explicitly.
- Serialised model access with bounded queueing (a single model isn't
  concurrency-safe) and backpressure
- Optional bearer-token auth
- Mock backend for CI / local dev; real backend behind one env var

Full API and protocol reference lives in [`docs/`](docs/).

## Quickstart (mock backend, no GPU)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Then, from another shell:

```bash
python client_examples/stream_client.py "Hello from a streaming server" out.wav
# or
bash client_examples/curl_examples.sh
```

## Running the real model

On a machine with a CUDA GPU:

```bash
pip install -r requirements.txt
pip install -U torch qwen-tts soundfile

export TTS_BACKEND=qwen
export TTS_DEVICE=cuda:0
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The only place that calls the `qwen-tts` library is
`QwenEngine._raw_stream` in `app/engine.py`. If the installed library version
uses a different streaming signature, adapt that one method — the rest of the
server (framing, streaming, HTTP) is backend-agnostic. If native streaming
isn't available, it automatically falls back to full generation re-chunked
into frames, so the HTTP API behaves identically.

## API

### `POST /v1/tts/stream` and `POST /v1/tts`

```jsonc
{
  "text": "Text to speak",          // required
  "model": "Qwen/Qwen3-TTS-...",    // optional; default model if omitted
  "language": "Auto",               // Auto | English | Chinese | Japanese | ...
  "mode": null,                     // null=infer; or custom_voice|voice_clone|voice_design
  "speaker": "Vivian",              // custom-voice mode
  "instruct": "a calm, warm tone",  // style modifier (custom voice) / design prompt
  "ref_audio": "https://.../a.wav", // voice-clone mode (optional)
  "ref_text": "transcript",         // required with ref_audio
  "response_format": "wav"          // "wav" | "pcm"
}
```

`GET /v1/models` lists the selectable model ids. `language` is validated against
the supported set (see `GET /v1/voices`); an unsupported value returns `422`.

- `wav` streams a WAV header followed by 16-bit PCM frames.
- `pcm` streams raw little-endian 16-bit mono PCM. The sample rate is returned
  in the `X-Sample-Rate` response header (24000 by default).

### `POST /v1/audio/speech` (OpenAI-compatible)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts", voice="Vivian", input="Hello!", response_format="wav",
) as resp:
    resp.stream_to_file("hello.wav")
```

### `WS /v1/tts/ws` (bidirectional streaming)

Send JSON control frames, receive binary audio frames concurrently. See
[`docs/websocket.md`](docs/websocket.md) for the full protocol, and
`client_examples/ws_client.py` for a runnable client. In short:

```jsonc
// client -> server
{"type":"config","model":"...","language":"English","speaker":"Vivian","response_format":"pcm"}
{"type":"synthesize","text":"Hello","request_id":"1"}   // overrides allowed
{"type":"cancel"}                                        // stop current synthesis
{"type":"close"}
// server -> client: {"type":"start",...} then binary audio, then {"type":"end",...}
```

### `GET /v1/models`, `GET /v1/voices`, `GET /health`

List selectable models, available speakers/languages, and check server status.

## Configuration

All settings are environment variables prefixed with `TTS_`; see
[`.env.example`](.env.example). Key ones:

| Variable | Default | Description |
|---|---|---|
| `TTS_BACKEND` | `mock` | `mock` or `qwen` |
| `TTS_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Default model |
| `TTS_MODELS` | *(empty)* | Extra selectable models (comma-separated allow-list) |
| `TTS_DEVICE` | `cuda:0` | Torch device for the real model |
| `TTS_SAMPLE_RATE` | `24000` | Output sample rate (Hz) |
| `TTS_STREAM_CHUNK_SAMPLES` | `1200` | Samples per streamed frame (~50 ms) |
| `TTS_MAX_QUEUE` | `32` | Max in-flight requests before `503` |
| `TTS_API_KEYS` | *(empty)* | Comma-separated bearer tokens; empty = no auth |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests run against the mock backend and are isolated from your local `.env`
(see `tests/conftest.py`), so they never load the real model or require a GPU.

## Docker

```bash
docker build -t qwen3-tts-server .
docker run --gpus all -p 8000:8000 -e TTS_BACKEND=qwen qwen3-tts-server
```

## Notes on streaming

- **WAV over a stream:** the streamed WAV header uses open-ended size fields
  (`0xFFFFFFFF`) since the length isn't known up front. Most players accept
  this; if a downstream tool is strict, use `response_format: "pcm"` and wrap
  it yourself, or use the non-streaming `/v1/tts` which writes exact sizes.
- **Latency:** `TTS_STREAM_CHUNK_SAMPLES` trades latency for overhead. Lower it
  for snappier first-audio; raise it to reduce per-frame cost.
- **Concurrency:** generation is serialised per process. To serve more load,
  run multiple workers/replicas behind a load balancer, each with its own GPU.

## Troubleshooting

Real issues hit while bringing this up (mostly Windows). See also
[`docs/deployment.md`](docs/deployment.md#troubleshooting).

### `ModuleNotFoundError: No module named 'torch'` on startup
Your `.env` has `TTS_BACKEND=qwen`, so the server tries to load the real model,
which imports `torch`. Either use the mock backend for local dev
(`TTS_BACKEND=mock`), or install the model deps (`pip install -U torch qwen-tts
soundfile`) on a CUDA machine.

### PowerShell: `Invoke-WebRequest : A positional parameter cannot be found …`
In PowerShell, `curl` is an **alias for `Invoke-WebRequest`** and doesn't accept
curl's `-s`/`-X`/`-d` flags. Use real curl as **`curl.exe`**, or native
PowerShell:

```powershell
$body = @{ text = "Hello there"; speaker = "Vivian" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/v1/tts" -Method Post `
  -ContentType "application/json" -Body $body -OutFile out.wav
```

If auth is enabled (`TTS_API_KEYS` set), add
`-Headers @{ Authorization = "Bearer <key>" }`. Note `Invoke-*` buffers the whole
response — for *true* streaming from a terminal use `curl.exe -N`, or the Python
client in `client_examples/`.

### `flash-attn` install fails: `CUDA_HOME environment variable is not set`
flash-attn compiles CUDA kernels at install time and needs the full CUDA
Toolkit (`nvcc`) — painful on Windows. **You don't need it.** Set
`TTS_ATTN_IMPLEMENTATION=sdpa` (PyTorch's built-in attention) and skip the
install. flash-attn is only an optional throughput optimisation.

### `torch.cuda.is_available()` returns `False`
You almost certainly installed the **CPU-only** wheel (`pip install torch` on
Windows defaults to CPU). Check with
`python -c "import torch; print(torch.__version__, torch.version.cuda)"` — a
`+cpu` version / `None` CUDA confirms it. Reinstall the CUDA build:

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Blackwell GPUs (RTX 50-series, `sm_120`) need cu128 or newer** — older
cu118/cu121 wheels lack the kernels and will error at runtime even if
`is_available()` is `True`. Pick the `cuXXX` index that matches (≤) your driver's
CUDA version (`nvidia-smi`, top-right).

### `500` — `model … does not support generate_voice_design`
The **CustomVoice** checkpoint has no voice-design mode; there `instruct` is a
**style modifier** on custom voice, not a separate mode. This server no longer
infers voice design from `instruct` — it folds `instruct` into `custom_voice`.
True voice design (a voice from description alone) needs the dedicated
**VoiceDesign** checkpoint and an explicit `"mode":"voice_design"`; requesting it
on a CustomVoice model returns `400`.

### Tests enforce auth / try to load the real model
Tests are isolated from your `.env` via `tests/conftest.py` (forces
`TTS_BACKEND=mock`, no auth). If you see `401`s or a slow model load during
`pytest`, make sure that file is present and you're not overriding `TTS_*` env
vars in your shell.
