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
- Three voice modes, selected automatically from the request:
  - **Custom voice** — a preset `speaker`
  - **Voice design** — a natural-language `instruct` prompt
  - **Voice cloning** — a `ref_audio` reference (path / URL / base64) + `ref_text`
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
  "speaker": "Vivian",              // custom-voice mode
  "instruct": "a calm, warm tone",  // voice-design mode (optional)
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
pip install pytest httpx
pytest -q
```

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
