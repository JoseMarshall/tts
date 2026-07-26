"""Tests for the audio helpers and the HTTP endpoints (mock backend)."""
from __future__ import annotations

import struct
import wave
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import float_to_pcm16, pcm_to_wav, wav_header
from app.engine import (
    DiaEngine,
    KokoroEngine,
    QwenEngine,
    available_backends,
    engine_class,
    kokoro_lang_code,
)
from app.main import app
from app.schemas import TTSRequest


# ---- voice-mode resolution ------------------------------------------------- #
def test_instruct_does_not_imply_voice_design():
    # instruct alone must fold into custom_voice (the CustomVoice model has no
    # voice_design); it must never auto-route to voice_design.
    req = TTSRequest(text="hi", instruct="a warm calm tone")
    assert req.resolve_mode() == "custom_voice"


def test_ref_audio_infers_voice_clone():
    req = TTSRequest(text="hi", ref_audio="ref.wav", ref_text="hello")
    assert req.resolve_mode() == "voice_clone"


def test_explicit_mode_wins():
    req = TTSRequest(text="hi", mode="voice_design", instruct="deep narrator")
    assert req.resolve_mode() == "voice_design"


# ---- backend registry & capabilities --------------------------------------- #
def test_backends_registered():
    for name in ("mock", "qwen", "kokoro", "dia"):
        assert name in available_backends()


def test_engine_class_unknown_raises():
    with pytest.raises(ValueError):
        engine_class("does-not-exist")


def test_qwen_capabilities():
    caps = QwenEngine.capabilities()
    assert caps["backend"] == "qwen"
    assert "Vivian" in caps["speakers"]


def test_kokoro_capabilities():
    caps = KokoroEngine.capabilities()
    assert caps["backend"] == "kokoro"
    assert "af_heart" in caps["speakers"]


def test_dia_capabilities():
    caps = DiaEngine.capabilities()
    assert caps["backend"] == "dia"
    assert caps["speakers"] == []          # dialogue tags, no preset voices
    assert DiaEngine.SAMPLE_RATE == 44100   # Dia outputs 44.1 kHz


# ---- multi-backend routing (EngineManager.resolve) ------------------------- #
def _manager():
    from app.config import Settings
    from app.manager import EngineManager
    # Default backend mock, plus qwen/kokoro/dia enabled for selection.
    return EngineManager(Settings(backend="mock", model_id="",
                                  backends="qwen,kokoro,dia", api_keys=""))


def test_resolve_by_backend_name():
    from app.manager import ModelSpec
    mgr = _manager()
    assert mgr.resolve("kokoro") == ModelSpec("kokoro", "hexgrad/Kokoro-82M")
    assert mgr.resolve("dia").backend == "dia"
    assert mgr.resolve("qwen").backend == "qwen"


def test_resolve_by_model_id():
    mgr = _manager()
    spec = mgr.resolve("hexgrad/Kokoro-82M")
    assert (spec.backend, spec.model_id) == ("kokoro", "hexgrad/Kokoro-82M")


def test_resolve_default_and_aliases():
    mgr = _manager()
    assert mgr.resolve(None) == mgr.default_spec
    assert mgr.resolve("default") == mgr.default_spec
    assert mgr.default_spec.backend == "mock"


def test_resolve_unknown_model_when_not_permissive():
    from app.config import Settings
    from app.manager import EngineManager, UnknownModelError
    # Default backend qwen (not mock) -> unknown ids are rejected.
    mgr = EngineManager(Settings(backend="qwen", model_id="qwen", backends=""))
    with pytest.raises(UnknownModelError):
        mgr.resolve("nari-labs/Dia-1.6B")   # dia not enabled here


def test_kokoro_lang_code_mapping():
    assert kokoro_lang_code("English") == "a"
    assert kokoro_lang_code("British English") == "b"
    assert kokoro_lang_code("Japanese") == "j"
    assert kokoro_lang_code("z") == "z"          # raw code passes through
    assert kokoro_lang_code(None) == "a"          # default
    assert kokoro_lang_code("Auto") == "a"
    with pytest.raises(ValueError):
        kokoro_lang_code("Klingon")


def test_speed_field_validation():
    assert TTSRequest(text="hi").speed == 1.0
    with pytest.raises(Exception):   # pydantic ValidationError: speed must be > 0
        TTSRequest(text="hi", speed=0)


# ---- audio helpers --------------------------------------------------------- #
def test_float_to_pcm16_roundtrip():
    x = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    pcm = float_to_pcm16(x)
    ints = np.frombuffer(pcm, dtype="<i2")
    assert ints.tolist() == [0, 32767, -32767, 16383]


def test_float_to_pcm16_clips():
    x = np.array([2.0, -2.0], dtype=np.float32)
    ints = np.frombuffer(float_to_pcm16(x), dtype="<i2")
    assert ints.tolist() == [32767, -32767]


def test_wav_header_sizes():
    hdr = wav_header(24000, data_size=2000)
    assert hdr[:4] == b"RIFF"
    assert struct.unpack("<I", hdr[4:8])[0] == 36 + 2000
    assert hdr[8:12] == b"WAVE"
    assert struct.unpack("<I", hdr[24:28])[0] == 24000  # sample rate


def test_pcm_to_wav_is_readable():
    pcm = float_to_pcm16(np.zeros(1000, dtype=np.float32))
    data = pcm_to_wav(pcm, 24000)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getnframes() == 1000


# ---- endpoints ------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client():
    # Context-manager form runs the lifespan (which builds the engine).
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_voices(client):
    # Reflects the active backend (mock here: permissive, empty lists).
    r = client.get("/v1/voices")
    body = r.json()
    assert body["backend"] == "mock"
    assert isinstance(body["speakers"], list)
    assert isinstance(body["languages"], list)


def test_non_streaming_wav(client):
    r = client.post("/v1/tts", json={"text": "hello world", "speaker": "Vivian"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() > 0


def test_streaming_wav_has_header_and_audio(client):
    r = client.post("/v1/tts/stream",
                    json={"text": "streaming please", "response_format": "wav"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"
    assert len(r.content) > 44  # header + at least one PCM frame


def test_streaming_pcm(client):
    r = client.post("/v1/tts/stream",
                    json={"text": "raw pcm", "response_format": "pcm"})
    assert r.status_code == 200
    assert r.headers["x-sample-rate"] == "24000"
    assert len(r.content) % 2 == 0  # whole 16-bit samples


def test_openai_endpoint(client):
    r = client.post("/v1/audio/speech",
                    json={"model": "qwen3-tts", "input": "hi", "voice": "Eric"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


# ---- model + language selection ------------------------------------------- #
def test_models_endpoint(client):
    r = client.get("/v1/models")
    body = r.json()
    assert body["object"] == "list"
    assert any(m["id"] == body["default"] for m in body["data"])


def test_request_selects_model_header(client):
    # Mock backend fabricates any model name; header echoes what was used.
    r = client.post("/v1/tts", json={"text": "pick me", "model": "my-custom-model"})
    assert r.status_code == 200
    assert r.headers["x-model"] == "my-custom-model"


def test_language_is_backend_specific(client):
    # Language validation is now the engine's job (mock accepts anything);
    # the global 422 no longer applies.
    ok = client.post("/v1/tts/stream", json={"text": "bonjour", "language": "French"})
    assert ok.status_code == 200
    anything = client.post("/v1/tts/stream",
                           json={"text": "hi", "language": "Elvish"})
    assert anything.status_code == 200  # mock is permissive


# ---- websocket ------------------------------------------------------------- #
def test_ws_bidirectional_stream(client):
    with client.websocket_connect("/v1/tts/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["default_model"]

        ws.send_json({"type": "config", "language": "English",
                      "speaker": "Vivian", "response_format": "pcm"})
        assert ws.receive_json()["type"] == "configured"

        ws.send_json({"type": "synthesize", "text": "hello over websocket",
                      "request_id": "abc"})
        started = ws.receive_json()
        assert started["type"] == "start" and started["request_id"] == "abc"
        assert started["sample_rate"] == 24000

        audio = bytearray()
        while True:
            frame = ws.receive()
            if "bytes" in frame and frame["bytes"] is not None:
                audio.extend(frame["bytes"])
            elif "text" in frame and frame["text"] is not None:
                import json
                ctrl = json.loads(frame["text"])
                if ctrl["type"] == "end":
                    assert ctrl["request_id"] == "abc"
                    break
        assert len(audio) > 0
        ws.send_json({"type": "close"})


def test_ws_unknown_message(client):
    with client.websocket_connect("/v1/tts/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "nonsense"})
        err = ws.receive_json()
        assert err["type"] == "error"
        ws.send_json({"type": "close"})


def test_ws_model_override(client):
    with client.websocket_connect("/v1/tts/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "synthesize", "text": "hi",
                      "model": "another-model", "request_id": "1"})
        start = ws.receive_json()
        assert start["type"] == "start"
        assert start["model"] == "another-model"
        ws.send_json({"type": "close"})
