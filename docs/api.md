# HTTP API Reference

Base URL: `http://<host>:<port>` (default `http://localhost:8000`).
All request bodies are JSON. Audio responses are binary.

If `TTS_API_KEYS` is set, send `Authorization: Bearer <key>` on every request.

---

## `POST /v1/tts/stream` — streaming synthesis

Streams audio using HTTP chunked transfer; bytes arrive as the model produces
them.

### Request body

| Field | Type | Default | Notes |
|---|---|---|---|
| `text` | string | — | **Required.** Text to synthesise. |
| `model` | string | server default | Selects the backend/model: a **backend name** (`qwen`, `kokoro`, `dia`) → that backend's default model, or a **model id** (`hexgrad/Kokoro-82M`). Must be selectable (`GET /v1/models`). Generic aliases (`default`, `auto`, `tts-1`) → the server default. |
| `language` | string | `Auto` | Backend-specific name/code (see `GET /v1/voices`). Unsupported values return `400` from the engine. |
| `mode` | string | *(inferred)* | Force a mode: `custom_voice` \| `voice_clone` \| `voice_design`. Omit to infer. |
| `speaker` | string | backend default | Preset voice (e.g. `Vivian` for Qwen, `af_heart` for Kokoro). |
| `speed` | number | `1.0` | Rate multiplier (0–4), for backends that support it (Kokoro). |
| `instruct` | string | — | Style modifier for custom voice, or the description for voice design (backend-dependent). |
| `ref_audio` | string | — | Reference audio (path/URL/base64) for cloning (backends that support it). |
| `ref_text` | string | — | Transcript of `ref_audio` (required with it). |
| `response_format` | string | `wav` | `wav` or `pcm`. |

**Voice mode** is inferred unless `mode` is set: `ref_audio` present → cloning;
otherwise → custom voice (with `instruct` as an optional style modifier).

`voice_design` — synthesising a voice from an `instruct` description with no
preset speaker — is **only** provided by the dedicated VoiceDesign checkpoint,
so it is never inferred. Request it explicitly with `"mode":"voice_design"`
while that model is selected. Asking a CustomVoice model for `voice_design`
returns `400`.

### Response

- Body: `audio/wav` (header + PCM frames) or `audio/pcm` (raw 16-bit LE mono).
- Headers: `X-Sample-Rate` (e.g. `24000`), `X-Model` (the model actually used).

### Example

```bash
curl -N -X POST http://localhost:8000/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there","model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
       "language":"English","speaker":"Vivian","response_format":"wav"}' \
  -o stream.wav
```

---

## `POST /v1/tts` — non-streaming synthesis

Same request body as above, but returns a single complete file. For `wav` the
container has **exact** size fields (unlike the streaming header).

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"A complete file.","speaker":"Eric"}' -o full.wav
```

---

## `POST /v1/audio/speech` — OpenAI-compatible

Accepts the OpenAI audio/speech schema and streams the body. Works with the
official OpenAI SDK.

### Request body

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | `qwen3-tts` | Mapped through the same alias/allow-list logic. |
| `input` | string | — | **Required.** Text (maps to `text`). |
| `voice` | string | `Vivian` | Maps to `speaker`. |
| `language` | string | `Auto` | Non-standard extension; honoured. |
| `instructions` | string | — | Maps to `instruct` (voice design). |
| `response_format` | string | `wav` | `wav` or `pcm`. |

### Example (OpenAI Python SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts", voice="Vivian", input="Hello!", response_format="wav",
) as resp:
    resp.stream_to_file("hello.wav")
```

---

## `GET /v1/models`

Lists selectable models (with their backend) in an OpenAI-shaped envelope. Each
`data[].id`, and each name in `backends`, is a valid value for a request's
`model` field.

```json
{
  "object": "list",
  "data": [
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "object": "model", "backend": "qwen"},
    {"id": "hexgrad/Kokoro-82M", "object": "model", "backend": "kokoro"},
    {"id": "nari-labs/Dia2-1B", "object": "model", "backend": "dia"}
  ],
  "default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "backends": ["dia", "kokoro", "qwen"]
}
```

## `GET /v1/voices`

Capabilities for a model's backend. Defaults to the default model; pass
`?model=<id-or-backend>` for a specific one (e.g. `?model=kokoro`).

```json
{
  "backend": "qwen",
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "speakers": ["Vivian", "Serena", "Uncle_Fu", "..."],
  "languages": ["Auto", "Chinese", "English", "..."],
  "default_speaker": "Vivian",
  "default_language": "Auto",
  "supports_marks": false,
  "supports_replicas": false
}
```

`supports_marks` tells you the backend emits word-level timing marks on the
WebSocket API (`true` for Kokoro) — see
[`websocket.md`](websocket.md#timing-marks-lip-sync).

`supports_replicas` is operator-facing: whether this backend may run several
concurrent instances via `TTS_ENGINE_REPLICAS`. It does not change the request
API — see [`configuration.md`](configuration.md#concurrency).

For `kokoro`, speakers are voices like `af_heart`/`bm_george`; for `dia`,
speakers are empty (use `[S1]`/`[S2]` tags). The `mock` backend returns empty
lists (accepts anything). An unknown `model` returns `400`.

## `GET /health`

```json
{
  "status": "ok",
  "backend": "mock",
  "available_backends": ["kokoro", "mock", "qwen"],
  "default_model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "models": ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"],
  "loaded": ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"],
  "sample_rate": 24000
}
```

---

## Speech-to-text endpoints

The server also transcribes. These take audio and return text; the WebSocket
equivalent (with turn detection) is
[`websocket.md`](websocket.md#speech-to-text-v1stt_ws).

### `POST /v1/stt` — native transcription

```jsonc
{"audio":"<base64 PCM/WAV/FLAC/MP3>",   // required
 "model":"whisper",                      // optional; backend name or model id
 "language":"en",                        // optional hint; omit to auto-detect
 "response_format":"text"}               // "text" (default) or "segments"
```

```json
{"text": "hello there"}
```

With `"response_format":"segments"` the response also carries a `segments` array.

### `POST /v1/stt/stream` — streaming transcription

Same body. Responds `application/x-ndjson`, one JSON object per line:

```jsonc
{"type":"start","model":"openai/whisper-large-v3"}
{"type":"segment","text":"hello there"}
{"type":"end"}
```

### `POST /v1/audio/transcriptions` — OpenAI-compatible

`multipart/form-data` with a `file` field, plus optional `model`, `language`
and `response_format`. Returns `text/plain` by default, or
`{"text": "..."}` for `response_format` of `json` / `verbose_json`.

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@speech.wav -F model=whisper -F response_format=json
```

### STT discovery

`GET /v1/stt/models` and `GET /v1/stt/voices[?model=…]` mirror their TTS
counterparts, listing selectable STT models and the active backend's languages
and accepted formats.

### STT status codes

`model` is checked against the operator's allow-list exactly as on the TTS side,
so an unlisted model is a `400` rather than a silent fallback.

| Code | Meaning |
|---|---|
| `400` | Unknown/-disallowed `model`; or `audio` present but empty or not valid base64; or a multipart request with no `file`. |
| `422` | The `audio` field is missing entirely — a schema violation rather than bad content. |
| `503` | Transcription queue full (`STT_MAX_QUEUE`). |

## Audio formats

| Format | Content-Type | Notes |
|---|---|---|
| `wav` (streaming) | `audio/wav` | Header has open-ended (`0xFFFFFFFF`) sizes; playable while streaming, but some strict tools read a bogus duration. |
| `wav` (`/v1/tts`) | `audio/wav` | Exact sizes; a normal finished file. |
| `pcm` | `audio/pcm` | Raw 16-bit little-endian mono; rate in `X-Sample-Rate`. Wrap it yourself if you need a container. |

**Tip:** for streaming into an audio pipeline, prefer `pcm` + `X-Sample-Rate`
and frame it yourself; use `/v1/tts` (`wav`) when you want a well-formed file.

## Status codes

| Code | Meaning |
|---|---|
| `200` | OK (streamed or complete). |
| `400` | Unknown/-disallowed `model`. |
| `401` | Missing/invalid API key (when auth enabled). |
| `422` | Validation error (missing `text`, unsupported `language`, …). |
| `503` | Server busy: generation queue full (`TTS_MAX_QUEUE`). |
