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

echo "# voice design via instruction"
curl -s -X POST "$BASE/v1/tts/stream" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Spooky narration.","instruct":"a slow, whispering horror voice"}' \
  -o design.wav
echo "wrote design.wav"

echo "# OpenAI-compatible endpoint"
curl -s -X POST "$BASE/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"OpenAI compatible route.","voice":"Serena"}' \
  -o openai.wav
echo "wrote openai.wav"
