# WebSocket API — bidirectional streaming

Two endpoints, in opposite directions:

| Endpoint | Direction | Section |
|---|---|---|
| `ws://<host>:<port>/v1/tts/ws` | text in → audio out | below |
| `ws://<host>:<port>/v1/stt_ws` | audio in → text out | [Speech-to-text](#speech-to-text-v1stt_ws) |

## Text-to-speech — `/v1/tts/ws`

The WebSocket lets a client **stream text segments and receive audio at the same
time**, over a single long-lived connection, and **cancel** an in-progress
utterance. It suits interactive/agentic use — e.g. piping an LLM's sentences in
as they're generated and playing audio back continuously.

- **Client → server** frames are **JSON text**.
- **Server → client** audio is sent as **binary** frames; control messages are
  **JSON text**. A client distinguishes them by frame type (bytes vs text).
- Send and receive run concurrently on the server, so control frames (like
  `cancel`) are handled while audio is streaming out.

> **Forward compatibility:** clients **must ignore unknown `type` values** (and
> unknown fields on known messages). New frame types — e.g. `marks` below — are
> added over time, and ignoring them is what lets features default to on
> without breaking older clients.

## Authentication

If `TTS_API_KEYS` is set, provide the key either as a header
(`Authorization: Bearer <key>`) or, for browsers, as a query parameter:
`ws://host/v1/tts/ws?api_key=<key>`. On failure the server closes with code
`1008`.

## Client → server messages

### `config` — set session defaults (optional)
```json
{"type":"config","model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
 "language":"English","speaker":"Vivian","response_format":"pcm"}
```
Fields: `model, language, speaker, instruct, ref_audio, ref_text,
response_format`. Persist for the connection until changed. Server replies
`{"type":"configured","session":{...}}`.

### `synthesize` — synthesise a segment
```json
{"type":"synthesize","text":"Hello there.","request_id":"1"}
```
Uses session defaults, and any of the same fields present here override them for
this segment. `request_id` is optional and echoed back on this segment's frames.
Only one synthesis runs at a time; sending another while one is active returns an
error (wait for its `end`).

### `cancel` — stop the active synthesis
```json
{"type":"cancel"}
```
Cooperative: generation stops at the next chunk boundary; the current segment
finishes with `{"type":"end","cancelled":true}`.

### `close` — end the session
```json
{"type":"close"}
```

## Server → client messages

| Message | When | Payload |
|---|---|---|
| `{"type":"ready","models":[...],"default_model":"..."}` | on connect | selectable models |
| `{"type":"configured","session":{...}}` | after `config` | effective defaults |
| `{"type":"start","request_id":...,"sample_rate":24000,"model":"...","format":"pcm","supports_marks":true}` | segment begins | stream metadata; `supports_marks` tells you whether this model emits timing marks |
| `{"type":"marks","request_id":...,"marks":[...]}` | during a segment, immediately before the audio frame whose range it covers | word-level timing marks (only when `supports_marks` and not disabled server-side) |
| *(binary frame)* | during a segment | audio: for `wav`, the first binary frame is the WAV header; then PCM frames. For `pcm`, raw 16-bit LE PCM frames. |
| `{"type":"end","request_id":...,"cancelled":false}` | segment done | `cancelled` true if stopped early |
| `{"type":"error","request_id":...,"message":"..."}` | on failure/bad input | socket stays open |

## Timing marks (lip sync)

Engines that compute alignment internally (currently **Kokoro**, English
pipelines) emit word-level timing marks as a byproduct of synthesis. Each
`marks` frame carries the marks whose time range falls inside the audio frame
sent immediately after it:

```jsonc
{"type":"marks","request_id":"3","marks":[
  {"kind":"word","text":"particularly","phonemes":"pɑɹˈtɪkjəlɚli","start":0.41,"end":1.02}
]}
```

- `start`/`end` are **seconds relative to the start of the segment**
  (`synthesize` request), matching the concatenated audio you receive — so the
  first audio sample of the segment is `t=0`.
- `text` is the word as written, `phonemes` its IPA phoneme string (mapping
  phonemes to mouth shapes/visemes is left to the client — every rig disagrees
  about the vocabulary).
- Only tokens that carry phonemes produce marks. Punctuation that occupies audio
  time as a pause has none, so it shows up as a gap between marks rather than as
  a mark of its own.
- Discovery without probing: `supports_marks` on the `start` frame (or
  `GET /v1/voices?model=...` on HTTP).
- Server operators can suppress the extra bytes with `TTS_EMIT_MARKS=0`.
- **Cancellation:** marks already sent describe audio you may never play if the
  segment is cancelled mid-stream — simply discard them with the audio.
- **WebSocket only.** HTTP chunked streaming keeps its current shape — there is
  no clean side channel in a chunked body, and every client that wants marks
  wants a socket anyway.

## Sequence

```
Client                         Server
  │───────── connect ─────────▶│
  │◀──────── ready ────────────│
  │──── config (defaults) ────▶│
  │◀────── configured ─────────│
  │─── synthesize {id:"1"} ───▶│
  │◀──────── start id=1 ───────│
  │◀──── marks (optional) ─────│
  │◀════ binary audio … ═══════│   (many frames, marks interleaved)
  │◀──────── end id=1 ─────────│
  │─── synthesize {id:"2"} ───▶│
  │◀──────── start id=2 ───────│
  │◀════ binary audio … ═══════│
  │──────── cancel ───────────▶│
  │◀── end id=2 cancelled ─────│
  │──────── close ────────────▶│
```

## Notes

- **Format:** `response_format:"pcm"` is easiest for real-time playback — each
  binary frame is directly appendable 16-bit LE PCM at the `sample_rate` from
  the `start` message. With `wav`, remember the first binary frame is the header.
- **Cancellation latency** depends on chunk size and model speed; with the mock
  backend you may receive a small burst already queued before it stops.
- **One-at-a-time:** to overlap synthesis, open multiple connections.
- **Disconnect:** if the client drops, the server cancels any active synthesis
  and cleans up the worker thread.

## Minimal client (Python)

See [`client_examples/ws_client.py`](../client_examples/ws_client.py) for a full
example. Core loop:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/v1/tts/ws") as ws:
        await ws.recv()  # ready
        await ws.send(json.dumps({"type":"config","language":"English",
                                  "response_format":"pcm"}))
        await ws.recv()  # configured
        await ws.send(json.dumps({"type":"synthesize","text":"Hi!","request_id":"1"}))
        pcm = bytearray()
        while True:
            f = await ws.recv()
            if isinstance(f, bytes):
                pcm.extend(f)
            elif json.loads(f)["type"] == "end":
                break
        await ws.send(json.dumps({"type":"close"}))

asyncio.run(main())
```

---

# Speech-to-text — `/v1/stt_ws`

Endpoint: `ws://<host>:<port>/v1/stt_ws`

The mirror image of `/v1/tts/ws`: the **client** sends audio and the **server**
sends text. Audio goes up as binary frames (or base64 inside a `chunk` message);
transcripts come back as JSON. The same forward-compatibility rule applies —
ignore `type` values you do not recognise.

Input audio is **16-bit little-endian mono PCM** at the `sample_rate` given in
the `ready` frame (`STT_SAMPLE_RATE`, 16000 by default).

## Client → server

| Message | Meaning |
|---|---|
| `{"type":"init","model":"whisper","language":"en","vad":{...}}` | session defaults; replies `configured` |
| `{"type":"start"}` | marks the beginning of a segment; replies `start` |
| `{"type":"chunk","data":"<base64 PCM>"}` | audio, base64-encoded |
| *(binary frame)* | audio, raw PCM — cheaper, prefer this |
| `{"type":"flush"}` | end the current turn now and transcribe it |
| `{"type":"cancel"}` | drop buffered audio and stop queued/running transcription |
| `{"type":"close"}` | end the session |

## Server → client

| Message | When |
|---|---|
| `{"type":"ready","models":[...],"default_model":"...","sample_rate":16000,"vad":{...}}` | on connect |
| `{"type":"configured","vad":{...}}` | after `init` |
| `{"type":"start","sample_rate":16000}` | after `start` |
| `{"type":"speech_start","t":1.28}` | VAD confirmed speech onset (only when VAD is on) |
| `{"type":"speech_end","t":3.94,"duration":2.66,"reason":"silence"}` | turn ended; transcription starting |
| `{"type":"segment","index":N,"text":"..."}` | a transcript segment, as the model produces it |
| `{"type":"done","count":N,"full_text":"...","reason":"silence","cancelled":false}` | turn complete |
| `{"type":"cancelled"}` | after `cancel` |
| `{"type":"error","message":"..."}` | failure or bad input; the socket stays open |

## Turn detection (hands-free, barge-in)

By default the server transcribes only when you send `flush` — you decide where
the utterance ends. That works for push-to-talk and for nothing else. Enable
server-side voice activity detection and the turn ends when the **speaker** stops:

```jsonc
{"type":"init","model":"whisper","vad":{"enabled":true,"silence_ms":700}}
```

From then on, `speech_start` fires when speech is confirmed and `speech_end`
fires after `silence_ms` of quiet, followed by the usual `segment` / `done`
frames. Those are byte-for-byte what an explicit `flush` always produced —
auto-flush is a new *trigger* for the existing pipeline, not a second one.

- **`speech_start` is the barge-in signal.** It is what tells a voice agent the
  user has started talking over the reply, so it can stop playback and `cancel`
  the in-flight `/v1/tts/ws` synthesis. The server does not do this for you: it
  does not know what has actually reached the speaker.
- **`reason`** on `speech_end` and `done` is `"silence"` (a natural ending),
  `"max_utterance"` (the hard cap fired — the turn was cut off mid-sentence), or
  `"client_flush"`.
- **`t`** is seconds since the session's first audio sample, so it correlates
  against your own playback clock across turns.
- **Explicit `flush` keeps working** with VAD on, and ends the current turn
  immediately.
- **Pre-roll:** the turn's audio starts ~`pre_roll_ms` *before* onset was
  confirmed, so the first phoneme is not clipped. Without it every transcript
  would start "…ello" instead of "Hello".

Per-session overrides on `init.vad` (each defaults to its `STT_VAD_*` setting):
`enabled`, `backend`, `threshold`, `speech_ms`, `silence_ms`, `pre_roll_ms`,
`max_utterance_s`. The `ready` frame reports the effective values plus
`available`, so a client can discover support without probing.

### Off by default

Timing marks default to *on* because they add an ignorable frame type. This is
different: auto-flush changes *when* transcription happens, which is session
semantics, and a client written against manual flushing would start receiving
`done` frames it never asked for. So it is opt-in per session — or set
`STT_VAD_AUTO_FLUSH=1` for a deployment where every client wants it.

If the configured detector cannot be loaded (e.g. `STT_VAD=silero` without
`pip install silero-vad`), the session gets one `error` frame and falls back to
manual `flush`. It does not lose the socket.

## Sequence (hands-free)

```
Client                         Server
  │───────── connect ─────────▶│
  │◀──────── ready ────────────│   vad: {available, enabled, silence_ms, …}
  │── init {vad:{enabled}} ───▶│
  │◀────── configured ─────────│
  │════ binary audio … ═══════▶│   (user starts talking)
  │◀────── speech_start ───────│   ← stop TTS playback here (barge-in)
  │════ binary audio … ═══════▶│
  │                            │   (user stops; silence_ms elapses)
  │◀─ speech_end reason=silence│
  │◀──────── segment 0 ────────│
  │◀──────── segment 1 ────────│
  │◀──────── done ─────────────│
  │════ binary audio … ═══════▶│   (next turn, same socket)
```

## Notes

- **Transcription runs off the receive loop.** Audio sent while a previous turn
  is still being transcribed is still read and detected on, which is what makes
  barge-in possible. A session's turns are transcribed in order.
- **Bounded buffering.** With VAD on, `max_utterance_s` caps how much audio a
  turn can accumulate. With VAD off the buffer grows until you `flush`.
- **`cancel`** drops buffered audio and signals queued and in-flight turns to
  stop; a turn already producing segments finishes with `"cancelled":true`.

---

## Browser client (JavaScript)

```js
const ws = new WebSocket("ws://localhost:8000/v1/tts/ws");
ws.binaryType = "arraybuffer";
const pcm = [];
ws.onmessage = (e) => {
  if (typeof e.data === "string") {
    const msg = JSON.parse(e.data);
    if (msg.type === "ready") {
      ws.send(JSON.stringify({type:"config", language:"English", response_format:"pcm"}));
      ws.send(JSON.stringify({type:"synthesize", text:"Hello from the browser", request_id:"1"}));
    } else if (msg.type === "end") {
      ws.send(JSON.stringify({type:"close"}));
      // feed `pcm` chunks into a Web Audio API AudioBuffer for playback
    }
  } else {
    pcm.push(new Int16Array(e.data)); // raw PCM frame
  }
};
```
