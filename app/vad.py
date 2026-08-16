"""Voice activity detection and turn endpointing for ``/v1/sst_ws``.

Two layers, deliberately separate:

* :class:`VAD` answers one narrow question — "is this one short frame speech?" —
  and nothing else. Backends register the way TTS engines do, so swapping
  detectors is a config change.
* :class:`TurnDetector` turns that per-frame answer into **turn boundaries**.
  Raw VAD output flickers; flushing on the first quiet frame chops the stop
  consonant off every word ending in /t/. All of the behaviour people actually
  tune (how much silence ends a turn, how much audio to keep from *before*
  speech was confirmed) lives here.

Adding a detector is self-contained::

    @vad_register
    class MyVAD(VAD):
        NAME = "mine"

        def __init__(self, sample_rate, **opts):
            super().__init__(sample_rate, **opts)
            self.frame_samples = ...     # window this detector demands
            ...                          # import heavy deps lazily, here

        def speech_prob(self, frame):
            return ...                   # 0.0 .. 1.0

Detectors bundled here:
* ``EnergyVAD``  — RMS + adaptive noise floor, no dependencies.
* ``SileroVAD``  — the ``silero-vad`` package (default; accurate, CPU, ~2 MB).
* ``WebrtcVAD``  — the ``webrtcvad`` package (what telephony pipelines use).
"""
from __future__ import annotations

import abc
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

log = logging.getLogger("tts.vad")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_VAD_REGISTRY: dict[str, type["VAD"]] = {}


def vad_register(cls: type["VAD"]) -> type["VAD"]:
    """Class decorator: register a detector under its ``NAME``."""
    if not getattr(cls, "NAME", None):
        raise ValueError(f"{cls.__name__} must define a NAME")
    _VAD_REGISTRY[cls.NAME] = cls
    return cls


def available_vads() -> list[str]:
    """Detector names that are *registered* (not necessarily installed)."""
    return sorted(_VAD_REGISTRY)


def vad_class(name: str) -> type["VAD"]:
    try:
        return _VAD_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown VAD {name!r}. Available: {available_vads()}"
        )


def build_vad(name: str, sample_rate: int, **opts) -> "VAD":
    """Construct a detector. Heavy imports happen here, not at module import,
    so the server still starts with no optional packages installed."""
    return vad_class(name)(sample_rate, **opts)


# --------------------------------------------------------------------------- #
# Detector interface
# --------------------------------------------------------------------------- #
class VAD(abc.ABC):
    """Per-frame speech detector.

    ``speech_prob`` rather than ``is_speech`` because Silero returns a
    probability, and discarding it to recover a threshold in the caller is a
    lossy round-trip. Detectors that only know yes/no return 0.0 or 1.0 and
    nothing downstream is worse for it.
    """

    NAME: str = ""
    #: Sample rates this detector accepts; empty means "any".
    SAMPLE_RATES: tuple[int, ...] = ()

    def __init__(self, sample_rate: int, **opts):
        if self.SAMPLE_RATES and sample_rate not in self.SAMPLE_RATES:
            raise ValueError(
                f"{self.NAME} VAD supports {self.SAMPLE_RATES} Hz, got {sample_rate}. "
                f"Resample the input or choose another SST_VAD."
            )
        self.sample_rate = sample_rate
        # Window size this detector wants, in samples. Subclasses set it.
        self.frame_samples: int = int(sample_rate * 0.03)  # 30 ms default

    @abc.abstractmethod
    def speech_prob(self, frame: np.ndarray) -> float:
        """P(speech) for exactly ``frame_samples`` of float32 mono audio."""

    def reset(self) -> None:
        """Clear any recurrent state. Called between turns."""


# --------------------------------------------------------------------------- #
# Energy detector — no dependencies
# --------------------------------------------------------------------------- #
@vad_register
class EnergyVAD(VAD):
    """RMS against an adaptive noise floor.

    Cheap, dependency-free and good enough for a close-talking mic in a quiet
    room. It is the reason the test suite and the ``mock`` path need nothing
    installed — the same role ``MockEngine`` plays for synthesis. It will
    embarrass itself on far-field or noisy input; that is what Silero is for.
    """

    NAME = "energy"

    def __init__(self, sample_rate: int, *, rms_floor: float = 0.005,
                 noise_ratio: float = 3.0, **opts):
        super().__init__(sample_rate, **opts)
        self.frame_samples = int(sample_rate * 0.03)
        self.rms_floor = rms_floor          # absolute gate: below this is never speech
        self.noise_ratio = noise_ratio      # speech must exceed N x the noise floor
        self._noise = 0.0                   # EMA of non-speech RMS

    def speech_prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        gate = max(self.rms_floor, self._noise * self.noise_ratio)
        if rms >= gate:
            return 1.0
        # Track the noise floor only on frames we judged silent, so a long
        # utterance can't drag the gate up above the speaker's own voice.
        self._noise = 0.95 * self._noise + 0.05 * rms if self._noise else rms
        return 0.0

    def reset(self) -> None:
        self._noise = 0.0


# --------------------------------------------------------------------------- #
# Silero VAD — the default
# --------------------------------------------------------------------------- #
@vad_register
class SileroVAD(VAD):
    """Silero VAD via the ``silero-vad`` package (``pip install silero-vad``).

    ~2 MB, runs on CPU well inside real time, and holds up on noisy and
    far-field input where an energy gate does not. It demands an exact window
    size (512 samples at 16 kHz, 256 at 8 kHz) — :class:`TurnDetector` does the
    re-framing.
    """

    NAME = "silero"
    SAMPLE_RATES = (8000, 16000)

    def __init__(self, sample_rate: int, **opts):
        super().__init__(sample_rate, **opts)
        self.frame_samples = 512 if sample_rate == 16000 else 256
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "SST_VAD=silero needs the 'silero-vad' package "
                "(pip install silero-vad torch). Set SST_VAD=energy for the "
                "zero-dependency detector."
            ) from exc

        self._torch = torch
        log.info("Loading Silero VAD (sample_rate=%d) ...", sample_rate)
        self._model = load_silero_vad()

    def speech_prob(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
            return float(self._model(tensor, self.sample_rate).item())

    def reset(self) -> None:
        reset = getattr(self._model, "reset_states", None)
        if reset is not None:
            reset()


# --------------------------------------------------------------------------- #
# WebRTC VAD
# --------------------------------------------------------------------------- #
@vad_register
class WebrtcVAD(VAD):
    """WebRTC's VAD via the ``webrtcvad`` package (``pip install webrtcvad``).

    Included because a lot of telephony pipelines already standardised on it,
    not because it beats Silero. Accepts only 10/20/30 ms frames of 16-bit PCM.
    """

    NAME = "webrtc"
    SAMPLE_RATES = (8000, 16000, 32000, 48000)

    def __init__(self, sample_rate: int, *, aggressiveness: int = 2, **opts):
        super().__init__(sample_rate, **opts)
        self.frame_samples = int(sample_rate * 0.03)  # 30 ms
        try:
            import webrtcvad
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "SST_VAD=webrtc needs the 'webrtcvad' package "
                "(pip install webrtcvad). Set SST_VAD=energy for the "
                "zero-dependency detector."
            ) from exc
        self._vad = webrtcvad.Vad(int(aggressiveness))

    def speech_prob(self, frame: np.ndarray) -> float:
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return 1.0 if self._vad.is_speech(pcm, self.sample_rate) else 0.0


# --------------------------------------------------------------------------- #
# Turn detection
# --------------------------------------------------------------------------- #
EndReason = Literal["silence", "max_utterance", "client_flush"]


@dataclass
class TurnEvent:
    """A turn boundary. ``t`` is seconds since the session's first sample."""

    kind: Literal["speech_start", "speech_end"]
    t: float
    duration: float = 0.0       # speech_end: seconds of audio in the turn
    reason: str = ""            # speech_end: why the turn ended
    audio: bytes = field(default=b"", repr=False)  # speech_end: the turn's PCM


class TurnDetector:
    """Wraps a :class:`VAD` with the hysteresis that makes it usable.

    Feed it the raw PCM16 byte stream in whatever sizes the client sends; it
    re-frames internally to the detector's window and emits
    ``speech_start`` / ``speech_end`` events.

    The state machine is two states and four knobs:

    * ``speech_ms``      consecutive speech before onset is believed — rejects
      clicks, keyboard taps and door slams.
    * ``silence_ms``     trailing silence that ends a turn. The one knob users
      will actually tune.
    * ``pre_roll_ms``    audio retained from *before* onset was confirmed.
      Without it every utterance starts "…ello" instead of "Hello", and the
      transcript is wrong in a way that reads as a model problem.
    * ``max_utterance_s`` hard cap. Also what bounds the buffer, so a client
      that streams and never flushes can no longer grow it until the process
      dies.
    """

    def __init__(
        self, vad: VAD, *, sample_rate: int, threshold: float = 0.5,
        speech_ms: int = 120, silence_ms: int = 700, pre_roll_ms: int = 300,
        max_utterance_s: float = 30.0,
    ):
        self.vad = vad
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_samples = vad.frame_samples
        self.frame_bytes = self.frame_samples * 2          # PCM16 mono
        self.frame_seconds = self.frame_samples / sample_rate

        def _frames(ms: int) -> int:
            return max(1, round((ms / 1000.0) / self.frame_seconds))

        self.speech_frames = _frames(speech_ms)
        self.silence_frames = _frames(silence_ms)
        self.pre_roll_frames = max(0, round((pre_roll_ms / 1000.0) / self.frame_seconds))
        self.max_frames = max(1, round(max_utterance_s / self.frame_seconds))

        # The pre-roll ring must also hold the frames that *confirm* onset,
        # or the confirming speech itself is dropped along with the silence.
        self._pre_roll: deque[bytes] = deque(
            maxlen=self.pre_roll_frames + self.speech_frames
        )
        self._turn: list[bytes] = []
        self._partial = bytearray()      # bytes not yet a whole frame
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._onset_pos = 0              # sample index where the current run began
        self._pos = 0                    # samples consumed this session

    # ---- introspection ---------------------------------------------------- #
    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def buffered_bytes(self) -> int:
        return sum(len(f) for f in self._turn) + len(self._partial)

    # ---- the loop --------------------------------------------------------- #
    def feed(self, pcm: bytes) -> list[TurnEvent]:
        """Consume PCM16 bytes; return any turn boundaries they crossed."""
        self._partial.extend(pcm)
        events: list[TurnEvent] = []
        while len(self._partial) >= self.frame_bytes:
            frame = bytes(self._partial[:self.frame_bytes])
            del self._partial[:self.frame_bytes]
            events.extend(self._feed_frame(frame))
        return events

    def _feed_frame(self, frame: bytes) -> list[TurnEvent]:
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        try:
            is_speech = self.vad.speech_prob(samples) >= self.threshold
        except Exception:  # a detector failure must not kill the session
            log.exception("VAD failed on a frame; treating it as silence.")
            is_speech = False

        self._pos += self.frame_samples
        events: list[TurnEvent] = []

        if not self._in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                if self._speech_run == 0:
                    # Remember where this run began — that, not the moment we
                    # became convinced, is when the speaker started talking.
                    self._onset_pos = self._pos - self.frame_samples
                self._speech_run += 1
            else:
                self._speech_run = 0

            if self._speech_run >= self.speech_frames:
                self._in_speech = True
                self._turn = list(self._pre_roll)   # pre-roll + confirming speech
                self._pre_roll.clear()
                self._silence_run = 0
                events.append(TurnEvent(
                    kind="speech_start", t=self._onset_pos / self.sample_rate,
                ))
            return events

        # ---- in speech ---------------------------------------------------- #
        self._turn.append(frame)
        self._silence_run = 0 if is_speech else self._silence_run + 1

        if self._silence_run >= self.silence_frames:
            events.append(self._end_turn("silence"))
        elif len(self._turn) >= self.max_frames:
            events.append(self._end_turn("max_utterance"))
        return events

    def flush(self) -> TurnEvent | None:
        """End the current turn now (an explicit client ``flush``).

        Mid-turn this returns the turn. Outside one it returns the retained
        pre-roll — an explicit flush is the client saying "transcribe what I
        sent", and an empty transcript answers that more honestly than an
        error does. The pre-roll is capped, so this stays bounded.

        Returns ``None`` only when genuinely nothing is buffered, so the
        handler can keep its existing "no audio data to transcribe" error.
        """
        pending = list(self._turn) if self._in_speech else list(self._pre_roll)
        if self._partial:
            # Don't strand a sub-frame tail on an explicit flush.
            pending.append(bytes(self._partial))
            self._partial.clear()
        if not pending:
            self._reset_state()
            return None
        self._turn = pending
        return self._end_turn("client_flush")

    def _end_turn(self, reason: EndReason) -> TurnEvent:
        audio = b"".join(self._turn)
        self._reset_state()
        return TurnEvent(
            kind="speech_end",
            t=self._pos / self.sample_rate,
            duration=len(audio) / 2 / self.sample_rate,
            reason=reason,
            audio=audio,
        )

    def _reset_state(self) -> None:
        self._turn = []
        self._pre_roll.clear()
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self.vad.reset()
