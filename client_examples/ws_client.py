"""Bidirectional WebSocket TTS client.

Streams several text segments over one connection and writes the received audio
to a WAV file as it arrives. Demonstrates the config/synthesize/cancel protocol.

    python client_examples/ws_client.py
"""
from __future__ import annotations

import asyncio
import json
import wave

import websockets

URL = "ws://localhost:8000/v1/tts/ws"  # add ?api_key=... if auth is enabled

SEGMENTS = [
    "Hello, this is the first segment.",
    "Here is a second sentence, streamed over the same socket.",
    "And a final one, in a chosen language and voice.",
]


async def main() -> None:
    async with websockets.connect(URL, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print("server ready:", ready)

        # Set session defaults: model + language + voice + raw PCM output.
        await ws.send(json.dumps({
            "type": "config",
            "model": ready["default_model"],
            "language": "English",
            "speaker": "Vivian",
            "response_format": "pcm",
        }))
        print("configured:", json.loads(await ws.recv())["type"])

        sample_rate = ready.get("sample_rate", 24000)
        pcm = bytearray()

        for i, text in enumerate(SEGMENTS):
            await ws.send(json.dumps({
                "type": "synthesize", "text": text, "request_id": str(i),
            }))
            # Consume until this request's "end" control frame.
            while True:
                frame = await ws.recv()
                if isinstance(frame, bytes):
                    pcm.extend(frame)
                    continue
                ctrl = json.loads(frame)
                if ctrl["type"] == "start":
                    sample_rate = ctrl["sample_rate"]
                    print(f"segment {i}: start (model={ctrl['model']}, "
                          f"supports_marks={ctrl.get('supports_marks')})")
                elif ctrl["type"] == "marks":
                    # Word timings for lip sync — ignore unknown frame types.
                    words = " ".join(m["text"] for m in ctrl["marks"])
                    print(f"segment {i}: marks [{words}]")
                elif ctrl["type"] == "end":
                    print(f"segment {i}: end, total {len(pcm)} bytes")
                    break
                elif ctrl["type"] == "error":
                    print("error:", ctrl["message"])
                    break

        await ws.send(json.dumps({"type": "close"}))

    with wave.open("ws_out.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))
    print("wrote ws_out.wav")


if __name__ == "__main__":
    asyncio.run(main())
