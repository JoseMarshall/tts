"""Stream TTS audio from the server and save it to a WAV file.

Demonstrates true streaming: bytes are written to disk as they arrive, so
playback could begin before synthesis finishes.

    python client_examples/stream_client.py "Hello there" out.wav
"""
from __future__ import annotations

import sys
import time

import requests

BASE = "http://localhost:8000"


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "Streaming text to speech, live."
    out = sys.argv[2] if len(sys.argv) > 2 else "out.wav"

    payload = {"text": text, "language": "Auto", "speaker": "Vivian",
               "response_format": "wav"}

    t0 = time.time()
    first_byte = None
    total = 0
    with requests.post(f"{BASE}/v1/tts/stream", json=payload, stream=True) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if first_byte is None:
                    first_byte = time.time() - t0
                    print(f"first audio byte after {first_byte*1000:.0f} ms")
                total += len(chunk)
                f.write(chunk)

    print(f"wrote {total} bytes to {out} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
