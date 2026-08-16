"""Tests for voice activity detection and turn endpointing on /v1/stt_ws.

Split into two halves on purpose:

* :class:`TurnDetector` is exercised with a *scripted* detector, so the
  hysteresis (onset confirmation, trailing silence, pre-roll, the utterance
  cap) is tested independently of whether any real VAD agrees.
* The WebSocket half runs the dependency-free ``energy`` detector end to end,
  because the thing worth protecting is that auto-flush produces exactly the
  frames an explicit flush always did.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import float_to_pcm16
from app.main import app
from app.vad import (
    VAD,
    EnergyVAD,
    TurnDetector,
    available_vads,
    build_vad,
    vad_class,
)

_SR = 16000


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Detector registry
# --------------------------------------------------------------------------- #
def test_vad_registry():
    assert {"energy", "silero", "webrtc"} <= set(available_vads())
    assert vad_class("energy") is EnergyVAD
    with pytest.raises(ValueError, match="Unknown VAD"):
        build_vad("nope", _SR)


def test_vad_rejects_unsupported_sample_rate():
    # Silero accepts 8k/16k only; the error must say so before anything loads.
    with pytest.raises(ValueError, match="supports"):
        build_vad("silero", 44100)


def test_energy_vad_silence_vs_tone():
    vad = build_vad("energy", _SR)
    assert vad.frame_samples == int(_SR * 0.03)
    silence = np.zeros(vad.frame_samples, dtype=np.float32)
    t = np.arange(vad.frame_samples, dtype=np.float32) / _SR
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert vad.speech_prob(silence) == 0.0
    assert vad.speech_prob(tone) == 1.0


# --------------------------------------------------------------------------- #
# Turn detection
# --------------------------------------------------------------------------- #
class _ScriptedVAD(VAD):
    """Returns a pre-baked probability per frame, so the state machine can be
    tested without real audio or a real model."""

    NAME = "scripted"

    def __init__(self, sample_rate, script=(), frame_samples=160):
        super().__init__(sample_rate)
        self.frame_samples = frame_samples      # 160 samples = 10 ms at 16 kHz
        self._script = list(script)
        self.resets = 0

    def speech_prob(self, frame):
        return self._script.pop(0) if self._script else 0.0

    def reset(self):
        self.resets += 1


def _frames_pcm(values, frame_samples=160):
    """One PCM16 frame per value, each frame filled with that value."""
    return [np.full(frame_samples, v, dtype="<i2").tobytes() for v in values]


def _detector(script, **kw):
    opts = dict(threshold=0.5, speech_ms=30, silence_ms=50,
                pre_roll_ms=20, max_utterance_s=1.0)
    opts.update(kw)
    return TurnDetector(_ScriptedVAD(_SR, script), sample_rate=_SR, **opts)


def test_turn_detector_frame_maths():
    det = _detector([])
    assert det.frame_samples == 160          # 10 ms
    assert det.speech_frames == 3            # 30 ms
    assert det.silence_frames == 5           # 50 ms
    assert det.pre_roll_frames == 2          # 20 ms
    assert det.max_frames == 100             # 1.0 s


def test_turn_detector_start_end_and_pre_roll():
    # 2 silent, 3 speech, 5 silent -> exactly one complete turn.
    det = _detector([0.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 5)
    pcm = _frames_pcm(range(1, 11))          # a distinct value per frame
    events = []
    for frame in pcm:
        events.extend(det.feed(frame))

    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    start, end = events

    # Onset is when the speech RUN began, not when it was finally believed.
    assert start.t == pytest.approx(0.02)
    assert end.reason == "silence"
    assert end.t == pytest.approx(0.10)      # 10 frames x 10 ms
    assert end.duration == pytest.approx(0.10)

    # The turn carries the pre-roll, so it starts before speech was confirmed.
    # Without this the transcript reads "...ello" instead of "Hello".
    assert end.audio == b"".join(pcm)
    assert np.frombuffer(end.audio, dtype="<i2")[0] == 1
    assert det.vad.resets == 1               # recurrent state cleared per turn


def test_turn_detector_pre_roll_is_capped():
    # 20 silent frames then speech: idle silence must not accumulate.
    det = _detector([0.0] * 20 + [1.0] * 3 + [0.0] * 5)
    for frame in _frames_pcm(range(1, 29)):
        for ev in det.feed(frame):
            if ev.kind == "speech_end":
                # 2 pre-roll + 3 speech + 5 silence = 10 frames, not 28.
                assert len(ev.audio) == 10 * 160 * 2
                return
    pytest.fail("no turn ended")


def test_turn_detector_max_utterance_forces_flush():
    # Speech that never stops must still end, which is what bounds the buffer.
    det = _detector([1.0] * 200)
    events = []
    for frame in _frames_pcm([7] * 150):
        events.extend(det.feed(frame))
    ends = [e for e in events if e.kind == "speech_end"]
    assert ends and ends[0].reason == "max_utterance"
    assert ends[0].duration == pytest.approx(1.0)   # max_utterance_s


def test_turn_detector_reframes_arbitrary_chunk_sizes():
    # The client's chunk sizes must not change what the detector sees.
    script = [0.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 5
    whole = b"".join(_frames_pcm(range(1, 11)))

    a = _detector(list(script))
    events_a = a.feed(whole)

    b = _detector(list(script))
    events_b = []
    for i in range(0, len(whole), 37):        # deliberately not frame-aligned
        events_b.extend(b.feed(whole[i:i + 37]))

    assert [e.kind for e in events_a] == [e.kind for e in events_b]
    assert [round(e.t, 6) for e in events_a] == [round(e.t, 6) for e in events_b]
    assert events_a[-1].audio == events_b[-1].audio


def test_turn_detector_flush_mid_turn_and_idle():
    det = _detector([1.0] * 3 + [0.0])
    for frame in _frames_pcm([5] * 4):
        det.feed(frame)
    assert det.in_speech
    ev = det.flush()
    assert ev is not None and ev.reason == "client_flush"
    assert not det.in_speech

    # Nothing buffered at all -> None, so the handler keeps its error frame.
    assert _detector([]).flush() is None


def test_turn_detector_flush_keeps_sub_frame_tail():
    det = _detector([])
    det.feed(b"\x01\x00" * 40)           # less than one 160-sample frame
    ev = det.flush()
    assert ev is not None and len(ev.audio) == 80


# --------------------------------------------------------------------------- #
# Over the WebSocket
# --------------------------------------------------------------------------- #
def _tone_pcm(seconds, freq=220.0, amp=0.5):
    n = int(_SR * seconds)
    t = np.arange(n, dtype=np.float32) / _SR
    return float_to_pcm16((amp * np.sin(2 * np.pi * freq * t)).astype(np.float32))


def _silence_pcm(seconds):
    return b"\x00\x00" * int(_SR * seconds)


def _drain_until(ws, wanted, limit=200):
    """Collect JSON frames until one of ``wanted`` types arrives."""
    seen = []
    for _ in range(limit):
        frame = ws.receive()
        if frame.get("text") is None:
            continue
        msg = json.loads(frame["text"])
        seen.append(msg)
        if msg["type"] in wanted:
            return seen
    pytest.fail(f"never saw {wanted}; got {[m['type'] for m in seen]}")


def test_stt_ws_ready_advertises_vad(client):
    with client.websocket_connect("/v1/stt_ws") as ws:
        ready = ws.receive_json()
        vad = ready["vad"]
        assert vad["available"] is True
        assert vad["enabled"] is False        # opt-in: it changes session semantics
        assert vad["backend"] == "silero"     # the configured default detector
        assert "silence_ms" in vad
        ws.send_json({"type": "close"})


def test_stt_ws_init_echoes_vad_config(client):
    with client.websocket_connect("/v1/stt_ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "init", "model": "mock",
                      "vad": {"enabled": True, "backend": "energy",
                              "silence_ms": 90}})
        conf = ws.receive_json()
        assert conf["type"] == "configured"
        assert conf["vad"]["enabled"] is True
        assert conf["vad"]["backend"] == "energy"
        assert conf["vad"]["silence_ms"] == 90
        ws.send_json({"type": "close"})


def test_stt_ws_vad_auto_flush(client):
    """Speech then silence transcribes with no explicit flush at all."""
    with client.websocket_connect("/v1/stt_ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "init", "model": "mock", "vad": {
            "enabled": True, "backend": "energy",
            "speech_ms": 60, "silence_ms": 90, "pre_roll_ms": 30,
        }})
        assert ws.receive_json()["type"] == "configured"

        ws.send_bytes(_tone_pcm(0.3))          # someone talks
        msgs = _drain_until(ws, {"speech_start"})
        assert msgs[-1]["type"] == "speech_start"

        ws.send_bytes(_silence_pcm(0.3))       # ... and stops
        msgs = _drain_until(ws, {"done"})
        kinds = [m["type"] for m in msgs]
        assert "speech_end" in kinds
        end = next(m for m in msgs if m["type"] == "speech_end")
        assert end["reason"] == "silence"      # not client_flush: nobody flushed
        assert end["duration"] > 0
        done = msgs[-1]
        assert done["type"] == "done" and done["reason"] == "silence"
        assert done["count"] > 0
        ws.send_json({"type": "close"})


def test_stt_ws_vad_max_utterance_bounds_the_buffer(client):
    """A client that talks forever and never flushes still gets cut off."""
    with client.websocket_connect("/v1/stt_ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "init", "model": "mock", "vad": {
            "enabled": True, "backend": "energy",
            "speech_ms": 60, "max_utterance_s": 0.2,
        }})
        ws.receive_json()

        ws.send_bytes(_tone_pcm(0.6))          # 3x the cap, no silence, no flush
        msgs = _drain_until(ws, {"done"})
        end = next(m for m in msgs if m["type"] == "speech_end")
        assert end["reason"] == "max_utterance"
        assert end["duration"] <= 0.25
        ws.send_json({"type": "close"})


def test_stt_ws_explicit_flush_still_works_with_vad_on(client):
    with client.websocket_connect("/v1/stt_ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "init", "model": "mock", "vad": {
            "enabled": True, "backend": "energy", "speech_ms": 60,
        }})
        ws.receive_json()
        ws.send_bytes(_tone_pcm(0.2))
        _drain_until(ws, {"speech_start"})

        ws.send_json({"type": "flush"})        # cut the turn short by hand
        msgs = _drain_until(ws, {"done"})
        end = next(m for m in msgs if m["type"] == "speech_end")
        assert end["reason"] == "client_flush"
        assert msgs[-1]["reason"] == "client_flush"
        ws.send_json({"type": "close"})


def test_stt_ws_vad_build_failure_falls_back_to_manual(client, monkeypatch):
    """A detector that will not load costs the session auto-flush, not the socket."""
    import app.main as main_mod

    def _boom(*a, **kw):
        raise RuntimeError("silero-vad not installed")

    monkeypatch.setattr(main_mod, "build_vad", _boom)

    with client.websocket_connect("/v1/stt_ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "init", "model": "mock", "vad": {"enabled": True}})
        ws.receive_json()

        ws.send_bytes(_tone_pcm(0.1))
        err = ws.receive_json()
        assert err["type"] == "error" and "VAD unavailable" in err["message"]

        # The session keeps working the old way.
        ws.send_bytes(_tone_pcm(0.1))
        ws.send_json({"type": "flush"})
        msgs = _drain_until(ws, {"done"})
        assert msgs[-1]["count"] > 0
        ws.send_json({"type": "close"})


def test_stt_ws_vad_off_by_default_is_unchanged(client):
    """Without opting in, audio is buffered until an explicit flush."""
    with client.websocket_connect("/v1/stt_ws") as ws:
        ws.receive_json()
        ws.send_bytes(_tone_pcm(0.3))
        ws.send_bytes(_silence_pcm(0.5))       # would auto-flush if VAD were on
        ws.send_json({"type": "flush"})
        msgs = _drain_until(ws, {"done"})
        # No VAD events at all, and the turn ends only because we asked.
        assert not [m for m in msgs if m["type"] in ("speech_start", "speech_end")]
        assert msgs[-1]["reason"] == "client_flush"
        ws.send_json({"type": "close"})


def test_stt_ws_cancel_drops_pending_audio(client):
    with client.websocket_connect("/v1/stt_ws") as ws:
        ws.receive_json()
        ws.send_bytes(_tone_pcm(0.2))
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["type"] == "cancelled"

        # The buffer was dropped, so there is nothing left to transcribe.
        ws.send_json({"type": "flush"})
        err = ws.receive_json()
        assert err["type"] == "error" and "no audio data" in err["message"]
        ws.send_json({"type": "close"})
