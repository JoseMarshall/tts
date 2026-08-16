"""Tests for the audio helpers and the HTTP endpoints (mock backend)."""
from __future__ import annotations

import json
import struct
import wave
import io
import base64 as _b64  # local alias for inline use

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import float_to_pcm16, pcm_to_wav, wav_header
from app.engine import (
    DiaEngine,
    KokoroEngine,
    Mark,
    MockEngine,
    QwenEngine,
    available_backends,
    engine_class,
    kokoro_lang_code,
    sst_available_backends,
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


def test_request_language_defaults_empty():
    # Empty so the engine/config default is used, not a hard-coded 'Auto'.
    assert TTSRequest(text="hi").language == ""


def test_engine_language_fallback_uses_config_default():
    from app.config import Settings
    from app.engine import MockEngine
    eng = MockEngine(Settings(default_language="Esperanto"))
    assert eng._language(TTSRequest(text="hi")) == "Esperanto"      # config default
    assert eng._language(TTSRequest(text="hi", language="French")) == "French"


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


# ---- timing marks ---------------------------------------------------------- #
class _FakeToken:
    """Minimal stand-in for misaki's MToken (word + optional timestamps)."""

    def __init__(self, text, phonemes, start_ts=None, end_ts=None):
        self.text = text
        self.phonemes = phonemes
        self.start_ts = start_ts
        self.end_ts = end_ts


class _FakeResult:
    """Minimal stand-in for kokoro's KPipeline.Result."""

    def __init__(self, audio, tokens=None):
        self.audio = audio
        self.tokens = tokens


class _FakePipeline:
    def __init__(self, results):
        self._results = list(results)

    def __call__(self, text, voice=None, speed=1.0):
        return iter(self._results)


def _kokoro_with(results, chunk=12000):
    """A KokoroEngine wired to a stub pipeline (no kokoro package, no model)."""
    from app.config import Settings
    from app.engine import TTSEngine
    eng = KokoroEngine.__new__(KokoroEngine)  # bypass __init__ (imports kokoro)
    TTSEngine.__init__(eng, Settings(stream_chunk_samples=chunk))
    eng._pipelines = {"a": _FakePipeline(results)}
    return eng


def test_supports_marks_capabilities():
    assert KokoroEngine.SUPPORTS_MARKS is True
    assert KokoroEngine.capabilities()["supports_marks"] is True
    for cls in (MockEngine, QwenEngine, DiaEngine):
        assert cls.SUPPORTS_MARKS is False
        assert cls.capabilities()["supports_marks"] is False


def test_voices_advertises_supports_marks(client):
    # Mock backend has no timings; kokoro's capability is on the class (above).
    assert client.get("/v1/voices").json()["supports_marks"] is False


def test_default_stream_marked_matches_stream():
    from app.config import Settings
    eng = MockEngine(Settings())
    req = TTSRequest(text="hello world")
    marked = list(eng.stream_marked(req))
    assert marked and all(marks == [] for _, marks in marked)
    np.testing.assert_array_equal(
        np.concatenate([c for c, _ in marked]),
        np.concatenate(list(eng.stream(req))),
    )


def test_kokoro_stream_marked_offsets():
    chunk = 12000  # samples -> 0.5 s frames at 24 kHz
    seg1 = np.full(2 * chunk, 0.1, dtype=np.float32)   # 1.0 s -> two frames
    seg3 = np.full(chunk, 0.2, dtype=np.float32)       # 0.5 s -> one frame
    eng = _kokoro_with([
        _FakeResult(seg1, tokens=[
            _FakeToken("Hello", "həlˈoʊ", 0.0, 0.5),
            _FakeToken("world", "wˈɜːld", 0.5, 1.0),
            _FakeToken(",", None, 0.9, 1.0),        # no phonemes -> skipped
            _FakeToken("um", "ʌm"),                  # untimed -> skipped
        ]),
        _FakeResult(None),                             # no audio: offset must not advance
        _FakeResult(seg3, tokens=[
            _FakeToken("again", "əɡˈɛn", 0.0, 0.5),
            _FakeToken("tail", "tˈeɪl", 0.625, 0.75),  # past seg end -> flushed
        ]),
    ])

    chunks = list(eng.stream_marked(TTSRequest(text="Hello world, um, again tail")))
    assert [len(c) for c, _ in chunks] == [chunk, chunk, chunk]

    # stream() must yield the exact same audio (single code path).
    np.testing.assert_array_equal(
        np.concatenate([c for c, _ in chunks]),
        np.concatenate(list(eng.stream(TTSRequest(text="x")))),
    )

    f0, f1, f2 = [marks for _, marks in chunks]
    assert [m.text for m in f0] == ["Hello"]
    assert [m.text for m in f1] == ["world"]
    assert [m.text for m in f2] == ["again", "tail"]

    # Times are rebased onto the request timeline across segments.
    assert (f0[0].start, f0[0].end) == (0.0, 0.5)
    assert (f1[0].start, f1[0].end) == (0.5, 1.0)
    assert (f2[0].start, f2[0].end) == (1.0, 1.5)    # after 1.0 s of seg1 audio
    assert (f2[1].start, f2[1].end) == (1.625, 1.75)  # 1.0 + segment-relative
    assert all(m.kind == "word" for m in f0 + f1 + f2)
    assert f0[0].phonemes == "həlˈoʊ"


def test_kokoro_stream_marked_ignores_untimed_results():
    # A segment with tokens=None (non-English pipelines) simply yields no marks.
    eng = _kokoro_with([_FakeResult(np.zeros(12000, dtype=np.float32))])
    chunks = list(eng.stream_marked(TTSRequest(text="quiet")))
    assert len(chunks) == 1 and chunks[0][1] == []


class _MarkedMockEngine(MockEngine):
    """Mock engine that also emits word marks (stands in for Kokoro)."""

    SUPPORTS_MARKS = True

    def stream_marked(self, req):
        step = self.settings.stream_chunk_samples
        sr = self.sample_rate
        marks = [
            Mark("word", "hello", "hɛlˈoʊ", 0.0, step / sr),
            Mark("word", "world", "wˈɜːld", 2 * step / sr, 3 * step / sr),
        ]
        for i in range(3):  # three frames of tone
            lo, hi = i * step / sr, (i + 1) * step / sr
            frame = np.full(step, 0.01 * (i + 1), dtype=np.float32)
            yield frame, [m for m in marks if lo <= m.start < hi]


def _patch_manager_get(monkeypatch, client, synth):
    async def _get(model=None):
        return synth
    monkeypatch.setattr(client.app.state.tts_manager, "get", _get)


def _collect_until_end(ws):
    """Collect WS frames until 'end'; 'audio' for binaries, else the JSON."""
    events = []
    while True:
        frame = ws.receive()
        if "bytes" in frame and frame["bytes"] is not None:
            events.append("audio")
        else:
            msg = json.loads(frame["text"])
            events.append(msg)
            if msg["type"] == "end":
                return events


def test_ws_marks(client, monkeypatch):
    from app.config import Settings
    from app.streaming import Synthesizer
    settings = Settings()
    synth = Synthesizer(_MarkedMockEngine(settings), settings)
    _patch_manager_get(monkeypatch, client, synth)

    with client.websocket_connect("/v1/tts/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "synthesize", "text": "hello world",
                      "request_id": "m1", "response_format": "pcm"})
        start = ws.receive_json()
        assert start["type"] == "start" and start["supports_marks"] is True

        events = _collect_until_end(ws)

    kinds = ["audio" if e == "audio" else e["type"] for e in events]
    # Each marks frame immediately precedes the audio frame it covers.
    assert kinds == ["marks", "audio", "audio", "marks", "audio", "end"]

    frames = [e for e in events if isinstance(e, dict) and e["type"] == "marks"]
    assert [f["request_id"] for f in frames] == ["m1", "m1"]
    assert frames[0]["marks"] == [
        {"kind": "word", "text": "hello", "phonemes": "hɛlˈoʊ",
         "start": 0.0, "end": 0.05}
    ]
    assert frames[1]["marks"][0]["text"] == "world"
    assert (frames[1]["marks"][0]["start"],
            frames[1]["marks"][0]["end"]) == (0.1, 0.15)


def test_ws_marks_disabled_by_settings(client, monkeypatch):
    from app.config import Settings
    from app.streaming import Synthesizer
    settings = Settings(emit_marks=False)  # TTS_EMIT_MARKS=0
    synth = Synthesizer(_MarkedMockEngine(settings), settings)
    _patch_manager_get(monkeypatch, client, synth)

    with client.websocket_connect("/v1/tts/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "synthesize", "text": "hello world",
                      "request_id": "m2", "response_format": "pcm"})
        start = ws.receive_json()
        assert start["supports_marks"] is True     # capability, not emission
        events = _collect_until_end(ws)

    kinds = ["audio" if e == "audio" else e["type"] for e in events]
    assert kinds == ["audio", "audio", "audio", "end"]


def test_ws_start_advertises_supports_marks(client):
    # The mock backend has no timings: start says so, and no marks are sent.
    with client.websocket_connect("/v1/tts/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "synthesize", "text": "hi", "request_id": "z",
                      "response_format": "pcm"})
        start = ws.receive_json()
        assert start["type"] == "start" and start["supports_marks"] is False
        events = _collect_until_end(ws)
        assert all(e == "audio" or e["type"] == "end" for e in events)
        ws.send_json({"type": "close"})


# ---- SST endpoint tests ---------------------------------------------------- #


def test_health_includes_sst(client):
    r = client.get("/health")
    body = r.json()
    assert body["status"] == "ok"
    assert "sst_default_model" in body
    assert "sst_enabled_backends" in body
    assert isinstance(body["sst_catalog"], list)


def test_sst_models_endpoint(client):
    r = client.get("/v1/sst/models")
    body = r.json()
    assert body["object"] == "list"
    assert "default" in body
    assert "backends" in body
    # mock is the only enabled backend by default.
    assert "mock" in body["backends"]


def test_sst_voices_endpoint(client):
    r = client.get("/v1/sst/voices")
    body = r.json()
    assert body["backend"] == "mock"
    assert isinstance(body["languages"], list)
    assert isinstance(body["supported_formats"], list)


def test_native_sst_endpoint(client):
    audio_bytes = b"\x00" * 320
    encoded = _b64.urlsafe_b64encode(audio_bytes).decode()

    r = client.post("/v1/sst", json={"audio": encoded})
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert len(body["text"]) > 0


def test_native_sst_endpoint_segments(client):
    audio_bytes = b"\x00" * 320
    encoded = _b64.urlsafe_b64encode(audio_bytes).decode()

    r = client.post("/v1/sst", json={"audio": encoded, "response_format": "segments"})
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert "segments" in body


def test_native_sst_endpoint_text(client):
    audio_bytes = b"\x00" * 320
    encoded = _b64.urlsafe_b64encode(audio_bytes).decode()

    r = client.post("/v1/sst", json={"audio": encoded, "response_format": "text"})
    assert r.status_code == 200
    body = r.json()
    assert "text" in body


def test_openai_sst_endpoint(client):
    """Verify the OpenAI-compatible /v1/audio/transcriptions endpoint accepts files."""
    # Create a minimal WAV audio file for the multipart upload.
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 320)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_buf.getvalue(), "audio/wav")},
        data={"model": "mock", "response_format": "text"},
    )
    assert r.status_code == 200


def test_openai_sst_endpoint_verbose(client):
    """OpenAI-compatible endpoint should return verbose_json when requested."""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 320)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_buf.getvalue(), "audio/wav")},
        data={"model": "mock", "response_format": "json"},
    )
    assert r.status_code == 200


def test_openai_sst_endpoint_no_file(client):
    """/v1/audio/transcriptions should reject requests without audio."""
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": "mock"},
    )
    assert r.status_code == 400


def test_streaming_sst_endpoint(client):
    audio_bytes = b"\x00" * 320
    encoded = _b64.urlsafe_b64encode(audio_bytes).decode()

    r = client.post("/v1/sst/stream", json={"audio": encoded})
    assert r.status_code == 200
    # NDJSON: "start\\n" ... segment lines ... "end\\n"
    text_body = r.content.decode()
    assert "start" in text_body.lower() or "segment" in text_body.lower()
    assert "end" in text_body.lower()


def test_sst_unknown_model(client):
    r = client.post("/v1/sst", json={"audio": "dGVzdA==", "model": "nonexistent"})
    assert r.status_code == 400


def test_sst_missing_audio_fields(client):
    # POST /v1/sst requires 'audio' field in the body.
    r = client.post("/v1/sst", json={"model": "mock"})
    assert r.status_code == 422


def test_sst_empty_audio(client):
    r = client.post("/v1/sst", json={"audio": ""})
    assert r.status_code == 400
    body = r.json()
    assert "audio" in body.get("detail", "").lower() or "required" in body.get("detail", "").lower()


def test_sst_stream_no_start_end(client):
    audio_bytes = b"\x00" * 320
    encoded = _b64.urlsafe_b64encode(audio_bytes).decode()

    r = client.post("/v1/sst/stream", json={"audio": encoded})
    text = r.content.decode()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    assert lines  # must have at least one frame


# ---- SST WebSocket tests --------------------------------------------------- #

def test_sst_ws_connect_and_ready(client):
    with client.websocket_connect("/v1/sst_ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert "models" in ready
        assert "default_model" in ready
        assert "sample_rate" in ready
        ws.send_json({"type": "close"})


def test_sst_ws_config_and_transcribe(client):
    with client.websocket_connect("/v1/sst_ws") as ws:
        # Connect → config → start → chunk → flush → done → close
        ready = ws.receive_json()
        assert ready["type"] == "ready"

        # Optional config
        ws.send_json({"type": "init", "model": "mock"})
        conf = ws.receive_json()
        assert conf["type"] == "configured"

        # Start segment
        ws.send_json({"type": "start"})
        start_ack = ws.receive_json()
        assert start_ack["type"] == "start"

        # Send audio chunk (base64)
        audio_bytes = b"\x00" * 320
        encoded = _b64.urlsafe_b64encode(audio_bytes).decode()
        ws.send_json({"type": "chunk", "data": encoded})

        # Flush → triggers transcription
        ws.send_json({"type": "flush"})

        # Receive segment messages and final done
        segments: list[dict] = []
        done_fired = False
        while True:
            frame = ws.receive()
            if "text" in frame:
                msg = json.loads(frame["text"])
                if msg["type"] == "done":
                    done_fired = True
                    break
                elif msg["type"] == "segment":
                    segments.append(msg)
        assert done_fired
        assert len(segments) > 0

        # Close
        ws.send_json({"type": "close"})


def test_sst_ws_binary_audio(client):
    """Client can send raw binary PCM instead of base64."""
    with client.websocket_connect("/v1/sst_ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "start"})

        # Send raw bytes (not base64) — server treats binary frames as PCM
        ws.send_bytes(b"\x00" * 320)
        ws.send_json({"type": "flush"})

        segments: list[dict] = []
        done_fired = False
        while True:
            frame = ws.receive()
            if "text" in frame:
                msg = json.loads(frame["text"])
                if msg["type"] == "done":
                    done_fired = True
                    break
                elif msg["type"] == "segment":
                    segments.append(msg)
        assert done_fired


def test_sst_ws_endpoint(client):
    """SST WebSocket endpoint handles sessions properly."""
    with client.websocket_connect("/v1/sst_ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        ws.send_json({"type": "close"})


def test_sst_ws_binary(client):
    """WS client can send raw binary PCM and get segments back."""
    with client.websocket_connect("/v1/sst_ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        ws.send_json({"type": "start"})
        start_ack = ws.receive_json()
        assert start_ack["type"] == "start"
        ws.send_bytes(b"\x00" * 320)
        ws.send_json({"type": "flush"})
        done_received = False
        for _ in range(20):
            frame = ws.receive()
            if "text" in frame:
                msg = json.loads(frame["text"])
                if msg["type"] == "done":
                    done_received = True
                    break
        assert done_received
