# WebSocket API — bidirectional streaming

Endpoint: `ws://<host>:<port>/v1/tts/ws`

The WebSocket lets a client **stream text segments and receive audio at the same
time**, over a single long-lived connection, and **cancel** an in-progress
utterance. It suits interactive/agentic use — e.g. piping an LLM's sentences in
as they're generated and playing audio back continuously.

- **Client → server** frames are **JSON text**.
- **Server → client** audio is sent as **binary** frames; control messages are
  **JSON text**. A client distinguishes them by frame type (bytes vs text).
- Send and receive run concurrently on the server, so control frames (like
  `cancel`) are handled while audio is streaming out.

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
| `{"type":"start","request_id":...,"sample_rate":24000,"model":"...","format":"pcm"}` | segment begins | stream metadata |
| *(binary frame)* | during a segment | audio: for `wav`, the first binary frame is the WAV header; then PCM frames. For `pcm`, raw 16-bit LE PCM frames. |
| `{"type":"end","request_id":...,"cancelled":false}` | segment done | `cancelled` true if stopped early |
| `{"type":"error","request_id":...,"message":"..."}` | on failure/bad input | socket stays open |

## Sequence

```
Client                         Server
  │───────── connect ─────────▶│
  │◀──────── ready ────────────│
  │──── config (defaults) ────▶│
  │◀────── configured ─────────│
  │─── synthesize {id:"1"} ───▶│
  │◀──────── start id=1 ───────│
  │◀════ binary audio … ═══════│   (many frames)
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
