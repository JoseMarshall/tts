#!/usr/bin/env bash
# Example requests against a running server (default: mock backend).
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"

echo "# health"
curl -s "$BASE/health"; echo

echo "# available voices"
curl -s "$BASE/v1/voices"; echo

echo "# streaming synthesis (custom voice) -> stream.wav"
curl -s -X POST "$BASE/v1/tts/stream" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Qwen three T T S.","speaker":"Vivian","response_format":"wav"}' \
  -o stream.wav
echo "wrote stream.wav"

echo "# non-streaming synthesis -> full.wav"
curl -s -X POST "$BASE/v1/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"A complete file in one response.","speaker":"Eric"}' \
  -o full.wav
echo "wrote full.wav"

echo "# custom voice with a style instruction (works on the CustomVoice model)"
curl -s -X POST "$BASE/v1/tts/stream" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Spooky narration.","speaker":"Dylan","instruct":"a slow, whispering horror voice"}' \
  -o instruct.wav
echo "wrote instruct.wav"

# Voice design (a voice from description alone, no preset speaker) needs the
# dedicated VoiceDesign checkpoint AND an explicit mode. It will 400 on the
# CustomVoice model:
#   -d '{"text":"...","mode":"voice_design","instruct":"a deep calm narrator"}'

# Kokoro backend example (run the server with TTS_BACKEND=kokoro):
#   curl -s -X POST "$BASE/v1/tts/stream" -H 'Content-Type: application/json' \
#     -d '{"text":"Hello from Kokoro.","speaker":"af_heart","language":"English","speed":1.0}' \
#     -o kokoro.wav

echo "# OpenAI-compatible endpoint"
curl -s -X POST "$BASE/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"OpenAI compatible route.","voice":"Serena"}' \
  -o openai.wav
echo "wrote openai.wav"
