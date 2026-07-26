# Multi-backend TTS Streaming Server

A FastAPI server that exposes text-to-speech models over HTTP with
**low-latency chunked streaming**. Audio starts flowing to the client as soon as
the model emits its first packet, rather than after the whole clip is generated.

**Pluggable backends** — pick one with `TTS_BACKEND`:

| Backend | Model | Notes |
|---|---|---|
| `mock` | — | Dependency-free tone generator. Runs anywhere; used for dev/CI. |
| `qwen` | [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | CUDA GPU. Custom voice / cloning / (design checkpoint). |
| `kokoro` | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Tiny (~350 MB), CPU or GPU. Preset voices, many languages. |
| `dia` | [Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) | GPU. Dialogue with `[S1]`/`[S2]` tags; 44.1 kHz; voice cloning. |

**One server can host several backends at once** and the client picks per request
(by backend name or model id) — see [Selecting a backend](#selecting-a-backend-per-request).
Adding another model is a self-contained ~40-line engine class plus one
decorator — see [Adding a backend](#adding-a-backend).

## Features

- `POST /v1/tts/stream` — chunked streaming synthesis (WAV or raw PCM)
- `POST /v1/tts` — full-file (non-streaming) synthesis
- `POST /v1/audio/speech` — **OpenAI-compatible** endpoint (works with the OpenAI SDK)
- `WS /v1/tts/ws` — **bidirectional WebSocket streaming** (send text/control,
  receive audio concurrently, cancel mid-utterance)
- **Pluggable backends** via an engine registry; `GET /v1/voices` and defaults
  are backend-aware
- **Client-selectable backend per request** — one server hosts several models
  (Qwen + Kokoro + Dia + …); the request's `model` picks by backend name or id
- **Per-request language, voice and speed** (`language`, `speaker`, `speed`);
  models are lazily loaded and restricted to an operator allow-list
- Voice modes (where the backend supports them):
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

## Running a real backend

### Qwen3-TTS (`qwen`) — CUDA GPU

```bash
pip install -r requirements.txt
pip install -U torch qwen-tts soundfile   # torch must match your CUDA

export TTS_BACKEND=qwen
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The only place that calls the `qwen-tts` library is `QwenEngine._raw_stream` in
`app/engine.py`; if a library update changes the streaming signature, adapt that
one method. Native streaming falls back to full-generation re-chunking if
unavailable, so the HTTP API is identical either way.

### Kokoro-82M (`kokoro`) — CPU or GPU

```bash
pip install -r requirements.txt
pip install -U kokoro soundfile
# espeak-ng is used for grapheme->phoneme (esp. non-English / OOV words):
#   Windows: winget install eSpeak-NG.eSpeak-NG
#   Debian/Ubuntu: sudo apt-get install espeak-ng
#   macOS: brew install espeak-ng

export TTS_BACKEND=kokoro
export TTS_MODEL_ID=hexgrad/Kokoro-82M
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Kokoro uses **preset voices** (`speaker`, e.g. `af_heart`, `bm_george`), a
`language` (name or single-char code — American English, `a`; Japanese, `j`; …),
and honours `speed`. It has no voice cloning/design, so those requests return
`400`. List everything with `GET /v1/voices`.

```bash
curl -s -X POST localhost:8000/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Kokoro.","speaker":"af_heart","language":"English","speed":1.0}' \
  -o kokoro.wav
```

### Dia-1.6B (`dia`) — GPU

```bash
pip install -r requirements.txt
pip install -U git+https://github.com/nari-labs/dia.git soundfile

export TTS_BACKEND=dia
export TTS_MODEL_ID=nari-labs/Dia-1.6B
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Dia is a **dialogue** model: mark speakers inline with `[S1]`/`[S2]` and add
non-verbals like `(laughs)`. It has no preset speakers; for a consistent voice,
clone one with `ref_audio` (+ `ref_text`). Output is 44.1 kHz.

```bash
curl -s -X POST localhost:8000/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"[S1] Hi there. [S2] Hello! (laughs)"}' -o dia.wav
```

## Selecting a backend per request

One server can host several backends and let the **client** choose per request.
Enable the extra backends the operator wants exposed, then select via `model`:

```bash
export TTS_BACKEND=qwen            # default backend
export TTS_BACKENDS=kokoro,dia     # also selectable by clients
uvicorn app.main:app
```

```bash
# by backend name (uses that backend's default model)
curl ... -d '{"text":"...","model":"kokoro","speaker":"af_heart"}'
curl ... -d '{"text":"[S1] Hi [S2] Hello","model":"dia"}'
# by explicit model id
curl ... -d '{"text":"...","model":"hexgrad/Kokoro-82M"}'
# omit "model" -> the default backend (qwen here)
```

`GET /v1/models` lists everything selectable (model ids + backend-name aliases);
`GET /v1/voices?model=<id-or-backend>` shows that model's voices/languages. Each
model loads lazily on first use, so exposing several costs nothing until used.
Unlisted models return `400` — so a request can't trigger an arbitrary download;
add specific ones with `TTS_MODELS=backend:model_id,...`.

## Adding a backend

Everything above the engine layer (routing, streaming, WebSocket, manager,
audio framing) is model-agnostic. A new backend is one registered `TTSEngine`
subclass in `app/engine.py`:

```python
@register
class MyEngine(TTSEngine):
    NAME = "mybackend"              # the TTS_BACKEND value that selects it
    SAMPLE_RATE = 24000
    SPEAKERS = ["..."]             # advertised by GET /v1/voices
    LANGUAGES = ["..."]
    DEFAULT_SPEAKER = "..."
    DEFAULT_LANGUAGE = "..."

    def __init__(self, settings, model_id=None):
        super().__init__(settings, model_id)
        ...                         # load the model; import heavy deps here

    def stream(self, req):          # the only required method
        for chunk in my_model.generate(req.text, voice=self._speaker(req)):
            yield chunk             # float32 mono numpy at SAMPLE_RATE
```

`stream()` is the sole primitive — non-streaming, WAV/PCM framing, WebSocket,
and cancellation all come for free. Raise `ValueError` for anything the model
can't do (it maps to `400`). The registry auto-discovers it via `@register`;
select it with `TTS_BACKEND=mybackend`.

## API

### `POST /v1/tts/stream` and `POST /v1/tts`

```jsonc
{
  "text": "Text to speak",          // required
  "model": "Qwen/Qwen3-TTS-...",    // optional; default model if omitted
  "language": "Auto",               // backend-specific name/code (see /v1/voices)
  "mode": null,                     // null=infer; or custom_voice|voice_clone|voice_design
  "speaker": "Vivian",              // preset voice; backend default if omitted
  "speed": 1.0,                     // rate multiplier (backends that support it)
  "instruct": "a calm, warm tone",  // style modifier (custom voice) / design prompt
  "ref_audio": "https://.../a.wav", // voice-clone mode (optional)
  "ref_text": "transcript",         // required with ref_audio
  "response_format": "wav"          // "wav" | "pcm"
}
```

`GET /v1/models` lists selectable model ids; `GET /v1/voices` lists the **active
backend's** speakers and languages. Speaker/language validity is backend-specific
— an unsupported value (or an unsupported `mode`) returns `400`.

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
| `TTS_BACKEND` | `mock` | **Default** engine: `mock`, `qwen`, `kokoro`, `dia`, … |
| `TTS_BACKENDS` | *(empty)* | Extra backends clients may select by name (comma-separated) |
| `TTS_MODEL_ID` | *(empty)* | Default model for the default backend (empty = backend's own default) |
| `TTS_MODELS` | *(empty)* | Extra selectable models, each `backend:model_id` (comma-separated allow-list) |
| `TTS_DEVICE` | `cuda:0` | Torch device (GPU backends) |
| `TTS_DEFAULT_SPEAKER` / `TTS_DEFAULT_LANGUAGE` | *(empty)* | Empty = backend's own default |
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

### Kokoro: `RuntimeError: espeak … not installed` or garbled non-English audio
Kokoro's grapheme→phoneme step uses **espeak-ng** (especially for non-English
and out-of-vocabulary words). Install the system package (`winget install
eSpeak-NG.eSpeak-NG` / `apt-get install espeak-ng` / `brew install espeak-ng`).
On Windows you may also need to point at the DLL, e.g.
`setx PHONEMIZER_ESPEAK_LIBRARY "C:\Program Files\eSpeak NG\libespeak-ng.dll"`.

### Kokoro: `400` on a request
Kokoro only does preset voices (`custom_voice`). `voice_clone` / `voice_design`
(via `ref_audio` or `mode`) aren't supported and return `400`. Also make sure
`speaker` is a valid Kokoro voice and `language` a supported name/code — see
`GET /v1/voices`.

### `Unknown TTS_BACKEND '…'` at startup
`TTS_BACKEND` must be a registered engine name — `mock`, `qwen`, or `kokoro`
(the error lists the available ones). Check for typos in `.env`/env vars.

### Tests enforce auth / try to load the real model
Tests are isolated from your `.env` via `tests/conftest.py` (forces
`TTS_BACKEND=mock`, no auth). If you see `401`s or a slow model load during
`pytest`, make sure that file is present and you're not overriding `TTS_*` env
vars in your shell.
