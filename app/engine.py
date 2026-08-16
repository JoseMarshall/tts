"""TTS engines.

An engine turns a :class:`~app.schemas.TTSRequest` into a stream of float32
audio chunks (mono, in [-1, 1], at ``engine.sample_rate``). Non-streaming
synthesis is just "consume the stream and concatenate", so ``stream`` is the
only primitive an engine must implement.

Adding a new backend is self-contained:

    @register
    class MyEngine(TTSEngine):
        NAME = "mybackend"          # value of TTS_BACKEND that selects it
        SAMPLE_RATE = 24000
        SPEAKERS = [...]            # advertised by GET /v1/voices
        LANGUAGES = [...]
        DEFAULT_SPEAKER = "..."
        DEFAULT_LANGUAGE = "..."

        def __init__(self, settings, model_id=None):
            super().__init__(settings, model_id)
            ...  # load the model (import heavy deps lazily, here)

        def stream(self, req):
            ...  # yield float32 mono numpy chunks

That's it — routing, streaming, WebSocket, the manager and audio framing are all
model-agnostic and need no changes. If the model also computes word/phoneme
timings, set ``SUPPORTS_MARKS = True`` and override ``stream_marked()`` so
WebSocket clients can receive timing marks alongside the audio.

Backends bundled here:
* ``MockEngine``   — dependency-free tone generator (no GPU / no downloads).
* ``QwenEngine``   — Qwen3-TTS via the ``qwen-tts`` package.
* ``KokoroEngine`` — Kokoro-82M via the ``kokoro`` package.
* ``DiaEngine``    — Dia-1.6B dialogue model via the ``dia`` package.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from .config import Settings
from .schemas import TTSRequest

log = logging.getLogger("tts.engine")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type["TTSEngine"]] = {}


def register(cls: type["TTSEngine"]) -> type["TTSEngine"]:
    """Class decorator: register an engine under its ``NAME``."""
    if not getattr(cls, "NAME", None):
        raise ValueError(f"{cls.__name__} must define a NAME")
    _REGISTRY[cls.NAME] = cls
    return cls


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def engine_class(name: str) -> type["TTSEngine"]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {available_backends()}"
        )


def engine_default_model(name: str) -> str:
    """The canonical model id for a backend (used when a client selects by
    backend name, e.g. model='kokoro')."""
    return engine_class(name).DEFAULT_MODEL or name


# --------------------------------------------------------------------------- #
# Timing marks
# --------------------------------------------------------------------------- #
@dataclass
class Mark:
    """A word (or phoneme) and when it occurs in the synthesized audio.

    ``start``/``end`` are seconds relative to the start of THIS request (the
    first emitted sample is t=0), so they line up with the concatenated audio
    the client receives. For ``kind == "word"``, ``text`` is the grapheme and
    ``phonemes`` that token's phoneme string; for ``kind == "phoneme"``,
    ``text`` is the symbol itself and ``phonemes`` is empty.
    """

    kind: Literal["word", "phoneme"]
    text: str
    phonemes: str
    start: float
    end: float


class TTSEngine(abc.ABC):
    # ---- backend metadata (override in subclasses) ------------------------ #
    NAME: str = ""
    DEFAULT_MODEL: str = ""                  # canonical model id for this backend
    SAMPLE_RATE: int | None = None          # None -> use settings.sample_rate
    SPEAKERS: list[str] = []
    LANGUAGES: list[str] = []
    DEFAULT_SPEAKER: str = ""
    DEFAULT_LANGUAGE: str = ""
    # True when the engine can emit word/phoneme timing marks (i.e. overrides
    # stream_marked). Advertised via capabilities() and GET /v1/voices.
    SUPPORTS_MARKS: bool = False
    # True when N instances of this backend can generate concurrently. Two
    # Python objects are not two independent models if they share a global
    # underneath — espeak-ng, a module-level cache, a single CUDA stream — and
    # that failure shows up as wrong output rather than a crash. So this is
    # opt-in per backend: TTS_ENGINE_REPLICAS>1 on a backend that has not
    # verified it runs with one instance and logs a warning.
    SUPPORTS_REPLICAS: bool = False

    def __init__(self, settings: Settings, model_id: str | None = None):
        self.settings = settings
        self.model_id = model_id or settings.model_id
        self.sample_rate = self.SAMPLE_RATE or settings.sample_rate

    @abc.abstractmethod
    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        """Yield successive float32 mono audio chunks for ``req``."""

    def stream_marked(
        self, req: TTSRequest
    ) -> Iterator[tuple[np.ndarray, list[Mark]]]:
        """Yield ``(audio chunk, marks)`` pairs: each chunk plus any marks
        whose time range falls inside it.

        The default costs nothing and never emits marks; engines that have
        timing information (``SUPPORTS_MARKS = True``) override this and
        define ``stream()`` in terms of it, so the two cannot drift.
        """
        for chunk in self.stream(req):
            yield chunk, []

    def synthesize(self, req: TTSRequest) -> np.ndarray:
        """Full (non-streaming) synthesis — concatenate the stream."""
        parts = list(self.stream(req))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def warmup(self) -> None:  # optional
        pass

    # ---- resolution helpers shared by subclasses -------------------------- #
    def _speaker(self, req: TTSRequest) -> str:
        return req.speaker or self.settings.default_speaker or self.DEFAULT_SPEAKER

    def _language(self, req: TTSRequest) -> str:
        return req.language or self.settings.default_language or self.DEFAULT_LANGUAGE

    @classmethod
    def capabilities(cls) -> dict:
        return {
            "backend": cls.NAME,
            "speakers": cls.SPEAKERS,
            "languages": cls.LANGUAGES,
            "default_speaker": cls.DEFAULT_SPEAKER,
            "default_language": cls.DEFAULT_LANGUAGE,
            "supports_marks": cls.SUPPORTS_MARKS,
            "supports_replicas": cls.SUPPORTS_REPLICAS,
        }


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #
@register
class MockEngine(TTSEngine):
    """Generates a short, pleasant chord whose length scales with text length.

    Useful to verify streaming, backpressure and client behaviour end-to-end
    without downloading a model or needing a GPU. Accepts any voice/language.
    """

    NAME = "mock"
    DEFAULT_MODEL = "mock"
    SUPPORTS_REPLICAS = True   # pure numpy, no shared state to corrupt

    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        sr = self.sample_rate
        # ~90 ms of audio per character, clamped to a sane range.
        seconds = min(max(len(req.text) * 0.09, 0.5), 30.0)
        total = int(seconds * sr)
        chunk = self.settings.stream_chunk_samples
        freqs = (220.0, 277.18, 329.63)  # A3 major-ish chord
        for start in range(0, total, chunk):
            n = min(chunk, total - start)
            t = (np.arange(start, start + n) / sr).astype(np.float32)
            wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
            # Gentle fade in/out over the whole clip to avoid clicks.
            env = np.minimum(t / 0.05, (seconds - t) / 0.05)
            env = np.clip(env, 0.0, 1.0).astype(np.float32)
            yield (0.3 * wave * env).astype(np.float32)


# --------------------------------------------------------------------------- #
# Qwen3-TTS backend
# --------------------------------------------------------------------------- #
@register
class QwenEngine(TTSEngine):
    NAME = "qwen"
    DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    SAMPLE_RATE = 24000
    DEFAULT_SPEAKER = "Vivian"
    DEFAULT_LANGUAGE = "Auto"
    SPEAKERS = [
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee",
    ]
    LANGUAGES = [
        "Auto", "Chinese", "English", "Japanese", "Korean", "German",
        "French", "Russian", "Portuguese", "Spanish", "Italian",
    ]

    def __init__(self, settings: Settings, model_id: str | None = None):
        super().__init__(settings, model_id)
        import torch  # noqa: F401 (validate availability early)
        from qwen_tts import Qwen3TTSModel

        dtypes = {
            "bfloat16": __import__("torch").bfloat16,
            "float16": __import__("torch").float16,
            "float32": __import__("torch").float32,
        }
        log.info("Loading %s on %s ...", self.model_id, settings.device)
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=settings.device,
            dtype=dtypes[settings.dtype],
            attn_implementation=settings.attn_implementation,
        )

    def warmup(self) -> None:
        try:
            list(self.stream(TTSRequest(text="Hello.")))
            log.info("Warmup complete.")
        except Exception:  # pragma: no cover - warmup is best-effort
            log.exception("Warmup failed (continuing anyway).")

    def _method_and_kwargs(self, req: TTSRequest):
        """Pick the qwen-tts method and kwargs for the request's mode."""
        mode = req.resolve_mode()
        lang = self._language(req)
        if mode == "voice_clone":
            return self.model.generate_voice_clone, dict(
                text=req.text, language=lang,
                ref_audio=req.ref_audio, ref_text=req.ref_text or "",
            )
        if mode == "voice_design":
            return self.model.generate_voice_design, dict(
                text=req.text, language=lang, instruct=req.instruct,
            )
        return self.model.generate_custom_voice, dict(
            text=req.text, language=lang,
            speaker=self._speaker(req), instruct=req.instruct or "",
        )

    def _raw_stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        """Call the library. THE one adaptation point for the real Qwen API.

        Tries native streaming (``stream=True``); if the installed version
        doesn't support it, falls back to a full generation and re-chunks it so
        callers still get a stream.
        """
        method, kwargs = self._method_and_kwargs(req)
        try:
            result = method(stream=True, **kwargs)
        except TypeError:
            result = None

        if result is not None and hasattr(result, "__iter__") \
                and not isinstance(result, tuple):
            for chunk in result:  # native streaming: iterator of chunks
                yield _as_float_mono(_chunk_audio(chunk))
            return

        # Fallback: generate the whole clip, then slice it into frames.
        wavs, sr = method(**kwargs)
        # Deliberately NOT `self.sample_rate = sr`: the WAV header and the
        # X-Sample-Rate response header were both written before this line
        # runs, so mutating it here would only make the engine disagree with
        # what the client was already told (and, with replicas, disagree with
        # its own siblings). Report the mismatch instead.
        if int(sr or 0) and int(sr) != self.sample_rate:
            log.warning(
                "%s returned %d Hz but the engine advertises %d Hz; audio will "
                "play at the wrong rate. Set TTS_SAMPLE_RATE=%d.",
                self.model_id, int(sr), self.sample_rate, int(sr),
            )
        audio = _as_float_mono(_first_wav(wavs))
        step = self.settings.stream_chunk_samples
        for start in range(0, len(audio), step):
            yield audio[start:start + step]

    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        for chunk in self._raw_stream(req):
            if chunk.size:
                yield chunk


# --------------------------------------------------------------------------- #
# Kokoro-82M backend
# --------------------------------------------------------------------------- #
# Kokoro selects language via a single-char code and speaks with preset voices
# whose prefix encodes language+gender (e.g. "af_heart" = American Female).
KOKORO_LANG_CODES = {
    "a": "a", "auto": "a", "english": "a", "american english": "a",
    "en": "a", "en-us": "a",
    "b": "b", "british english": "b", "en-gb": "b",
    "e": "e", "spanish": "e", "es": "e",
    "f": "f", "french": "f", "fr": "f", "fr-fr": "f",
    "h": "h", "hindi": "h", "hi": "h",
    "i": "i", "italian": "i", "it": "i",
    "j": "j", "japanese": "j", "ja": "j",
    "p": "p", "brazilian portuguese": "p", "portuguese": "p", "pt-br": "p",
    "z": "z", "mandarin chinese": "z", "chinese": "z", "mandarin": "z", "zh": "z",
}


def kokoro_lang_code(language: str | None, default: str = "a") -> str:
    """Map a friendly language name or code to a Kokoro single-char lang code."""
    key = (language or "").strip().lower()
    if key in ("", "auto"):
        return default
    code = KOKORO_LANG_CODES.get(key)
    if code is None:
        raise ValueError(
            f"Unsupported language {language!r} for Kokoro. Use a code "
            f"(a,b,e,f,h,i,j,p,z) or a name like English/Spanish/Japanese/…"
        )
    return code


@register
class KokoroEngine(TTSEngine):
    NAME = "kokoro"
    DEFAULT_MODEL = "hexgrad/Kokoro-82M"
    SAMPLE_RATE = 24000
    DEFAULT_SPEAKER = "af_heart"
    DEFAULT_LANGUAGE = "a"
    SUPPORTS_MARKS = True
    LANGUAGES = [
        "American English", "British English", "Spanish", "French", "Hindi",
        "Italian", "Japanese", "Brazilian Portuguese", "Mandarin Chinese",
    ]
    # A representative subset; the model card lists the full voice inventory.
    SPEAKERS = [
        "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
        "am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_adam",
        "bf_emma", "bf_isabella", "bf_alice", "bm_george", "bm_fable", "bm_lewis",
        "ef_dora", "em_alex", "ff_siwis", "hf_alpha", "hm_omega",
        "if_sara", "im_nicola", "jf_alpha", "jm_kumo", "pf_dora", "pm_alex",
        "zf_xiaobei", "zm_yunjian",
    ]

    def __init__(self, settings: Settings, model_id: str | None = None):
        super().__init__(settings, model_id)
        from kokoro import KPipeline  # heavy import, kept local

        self._KPipeline = KPipeline
        self._pipelines: dict[str, object] = {}  # one per lang code

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            log.info("Building Kokoro pipeline for lang_code=%r", lang_code)
            self._pipelines[lang_code] = self._KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

    def warmup(self) -> None:
        try:
            list(self.stream(TTSRequest(text="Hello.")))
            log.info("Warmup complete.")
        except Exception:  # pragma: no cover
            log.exception("Warmup failed (continuing anyway).")

    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        # Audio-only view of stream_marked(): one code path, so the two can
        # never drift.
        for chunk, _marks in self.stream_marked(req):
            yield chunk

    def stream_marked(
        self, req: TTSRequest
    ) -> Iterator[tuple[np.ndarray, list[Mark]]]:
        mode = req.resolve_mode()
        if mode != "custom_voice":
            raise ValueError(
                "Kokoro supports only preset voices (custom_voice); "
                "voice_clone / voice_design are not available on this backend."
            )
        lang_code = kokoro_lang_code(self._language(req), self.DEFAULT_LANGUAGE)
        voice = self._speaker(req)
        speed = req.speed or 1.0
        pipeline = self._pipeline(lang_code)
        step = self.settings.stream_chunk_samples
        sr = self.sample_rate

        # Kokoro yields per text segment (naturally streaming). Re-slice each
        # segment into fixed frames for smooth, consistent delivery downstream.
        #
        # Marks come free: KPipeline.join_timestamps writes start_ts/end_ts
        # (seconds, relative to the SEGMENT) onto each token, using the same
        # pred_dur the audio was generated from — and already scaled by
        # `speed` inside the model, so no rate correction is needed here.
        # Non-English pipelines yield tokens=None and simply produce no marks.
        offset = 0  # samples emitted so far, across all segments
        for result in pipeline(req.text, voice=voice, speed=speed):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            if audio is None:      # segment produced no audio (e.g. a pause)
                continue           # ... so the sample offset must not advance
            audio = _as_float_mono(audio)

            # Rebase the segment's word marks onto the request timeline.
            seg_base = offset / sr
            pending = [
                Mark(
                    kind="word",
                    text=getattr(t, "text", "") or "",
                    phonemes=t.phonemes,
                    start=seg_base + float(t.start_ts),
                    end=seg_base + float(t.end_ts),
                )
                for t in (getattr(result, "tokens", None) or [])
                if getattr(t, "phonemes", None)
                and getattr(t, "start_ts", None) is not None
                and getattr(t, "end_ts", None) is not None
            ]

            for s in range(0, len(audio), step):
                frame = audio[s:s + step]
                if not frame.size:
                    continue
                frame_end = (offset + frame.size) / sr
                # Emit each mark with the frame whose range contains its
                # start. On the segment's last frame, flush whatever remains:
                # rounding in join_timestamps can push a final token's start
                # just past the end of the segment's audio.
                last = s + step >= len(audio)
                out = [m for m in pending if m.start < frame_end or last]
                pending = [m for m in pending
                           if m.start >= frame_end and not last]
                offset += frame.size
                yield frame, out


# --------------------------------------------------------------------------- #
# Dia-1.6B backend (nari-labs)
# --------------------------------------------------------------------------- #
@register
class DiaEngine(TTSEngine):
    """Dia is a dialogue TTS model: speakers are marked inline with [S1]/[S2]
    tags (and non-verbals like ``(laughs)``), so there are no preset voices.
    It supports voice cloning via an audio prompt, outputs 44.1 kHz, and is not
    natively streaming — we generate the clip and re-chunk it into frames.
    """

    NAME = "dia"
    DEFAULT_MODEL = "nari-labs/Dia-1.6B"
    SAMPLE_RATE = 44100
    DEFAULT_LANGUAGE = "English"
    LANGUAGES = ["English"]
    SPEAKERS: list[str] = []   # none; use [S1]/[S2] tags in the text

    def __init__(self, settings: Settings, model_id: str | None = None):
        super().__init__(settings, model_id)
        from dia.model import Dia  # heavy import, kept local

        # Dia takes a string compute dtype; our settings.dtype already matches.
        self.model = Dia.from_pretrained(
            self.model_id, compute_dtype=settings.dtype or "float16"
        )

    def warmup(self) -> None:
        try:
            list(self.stream(TTSRequest(text="[S1] Hello. [S2] Hi there.")))
            log.info("Warmup complete.")
        except Exception:  # pragma: no cover
            log.exception("Warmup failed (continuing anyway).")

    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        mode = req.resolve_mode()
        if mode == "voice_design":
            raise ValueError(
                "Dia has no voice_design mode. Use [S1]/[S2] dialogue tags, "
                "or voice_clone with a reference audio prompt."
            )

        text = req.text
        kwargs: dict = {}
        if mode == "voice_clone":
            if not req.ref_audio:
                raise ValueError("voice_clone requires 'ref_audio' for Dia.")
            kwargs["audio_prompt"] = req.ref_audio
            # Dia clones best when the reference transcript prefixes the target.
            if req.ref_text:
                text = f"{req.ref_text}\n{req.text}"

        audio = self.model.generate(text, verbose=False, **kwargs)
        audio = _as_float_mono(audio)
        step = self.settings.stream_chunk_samples
        for s in range(0, len(audio), step):
            frame = audio[s:s + step]
            if frame.size:
                yield frame


# --------------------------------------------------------------------------- #
# Normalisation helpers (tolerant of the shapes/types libraries return)
# --------------------------------------------------------------------------- #
def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):        # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _first_wav(wavs) -> np.ndarray:
    arr = _to_numpy(wavs)
    if arr.ndim >= 2:  # batch of clips -> take the first
        arr = arr[0]
    return arr


def _chunk_audio(chunk) -> np.ndarray:
    # A streaming chunk may be a bare array or a (array, sr) tuple.
    if isinstance(chunk, tuple):
        chunk = chunk[0]
    return _to_numpy(chunk)


def _as_float_mono(arr) -> np.ndarray:
    arr = _to_numpy(arr)
    if arr.ndim > 1:  # (channels, n) or (n, channels) -> mono
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[-1] else 1)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
def build_engine(settings: Settings, model_id: str | None = None) -> TTSEngine:
    return engine_class(settings.backend)(settings, model_id)


# --------------------------------------------------------------------------- #
# STT (Speech-to-text) engines — parallel hierarchy                           #
# --------------------------------------------------------------------------- #

_STT_REGISTRY: dict[str, type["SSEngine"]] = {}


def _stt_register(cls: type["SSEngine"]) -> type["SSEngine"]:
    """Class decorator: register an STT engine under its ``NAME``."""
    if not getattr(cls, "NAME", None):
        raise ValueError(f"{cls.__name__} must define a NAME")
    _STT_REGISTRY[cls.NAME] = cls
    return cls


def stt_available_backends() -> list[str]:
    return sorted(_STT_REGISTRY)


def stt_engine_class(name: str) -> type["SSEngine"]:
    try:
        return _STT_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown STT backend {name!r}. Available: {stt_available_backends()}"
        )


def stt_engine_default_model(name: str) -> str:
    return stt_engine_class(name).DEFAULT_MODEL or name


class SSEngine(abc.ABC):
    """Abstract base for all speech-to-text engines.

    An STT engine takes audio bytes and produces text.  Streaming engines can
    yield partial segments while a non-streaming one waits for the complete
    result before returning.

    Adding a new STT backend mirrors the TTS pattern — register, define metadata,
    implement ``__init__`` (heavy imports here) and ``stream_transcribe()``. No
    changes are needed in routing, framing or the manager.

    Backends bundled here:
    * ``MockSSEngine``   -- dependency-free placeholder that echoes back the first
      few words of a synthetic transcript.
    * ``VoxtralEngine``  -- mistralai/Voxtral-Small-24B via transformers.
    * ``WhisperEngine``  -- openai/whisper-large-v3 (or tiny/small/…) via native
      whisper library or the ``transformers`` pipeline.
    * ``OpenAIEngine``   -- delegates to OpenAI's /v1/audio/transcriptions API.
    """

    # ---- backend metadata (override in subclasses) ------------------------ #
    NAME: str = ""
    DEFAULT_MODEL: str = ""
    SAMPLE_RATE: int | None = None          # target sample rate for resampling
    MAX_INPUT_SECONDS: float = 30.0         # reject longer audio to guard GPU
    LANGUAGES: list[str] = []              # hints the model accepts
    SUPPORTED_FORMATS: frozenset[str] = frozenset({"wav", "flac", "mp3", "pcm"})

    def __init__(self, settings: "Settings", model_id: str | None = None):
        self.settings = settings
        self.model_id = model_id or (
            getattr(settings, "stt_model_id", "") or stt_engine_default_model(self.NAME)
        )

    @abc.abstractmethod
    def stream_transcribe(
        self, audio_bytes: bytes, *, language: str | None = None,
    ) -> Iterator[str]:
        """Yield partial text segments as the model produces them."""

    def transcribe(
        self, audio_bytes: bytes, *, language: str | None = None,
    ) -> str:
        """Full (non-streaming) transcription — concatenate all segments."""
        segments = list(self.stream_transcribe(audio_bytes, language=language))
        return " ".join(s for s in segments if s.strip())

    def warmup(self) -> None:          # optional; best-effort
        pass

    # ---- input decoding (shared by every STT backend) --------------------- #
    @property
    def target_sample_rate(self) -> int:
        return self.SAMPLE_RATE or getattr(self.settings, "sample_rate", 16000)

    @property
    def max_input_samples(self) -> int:
        seconds = float(
            getattr(self.settings, "max_input_seconds", 0) or self.MAX_INPUT_SECONDS
        )
        return int(self.target_sample_rate * seconds)

    def decode(self, audio_bytes: bytes) -> np.ndarray:
        """Request bytes -> float32 mono samples at ``target_sample_rate``,
        truncated to ``max_input_seconds``."""
        return decode_audio(audio_bytes, self.target_sample_rate)[
            : self.max_input_samples
        ]

    def _language(self, req_language: str | None) -> str | None:
        """Resolve a language hint, falling through settings → backend default."""
        # Accept either the TTS-style or STT-style field name for compatibility.
        return (req_language
                or getattr(self.settings, "default_language", None)
                or getattr(self.settings, "stt_default_language", None))

    @classmethod
    def capabilities(cls) -> dict:
        return {
            "backend": cls.NAME,
            "languages": cls.LANGUAGES,
            "default_model": getattr(cls, "DEFAULT_MODEL", ""),
            "sample_rate": getattr(cls, "SAMPLE_RATE", None),
            "supported_formats": sorted(cls.SUPPORTED_FORMATS),
        }


# --------------------------------------------------------------------------- #
# STT input decoding                                                          #
# --------------------------------------------------------------------------- #
def decode_audio(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode request audio to float32 mono at ``target_sr``.

    The STT engines are fed from two callers carrying different bytes:
    ``/v1/stt`` and ``/v1/audio/transcriptions`` forward a whole uploaded
    **file** (wav/mp3/flac/…), while ``/v1/stt_ws`` streams **raw
    little-endian int16 PCM** frames. Both arrive as bare ``bytes`` with
    nothing to tell them apart, so try a container first and fall back to raw
    PCM16 — a headerless buffer is exactly what soundfile rejects.

    Never hand these bytes straight to a transformers pipeline: it treats
    ``bytes`` as an encoded file and shells out to ffmpeg, which fails on raw
    PCM with "Soundfile is either not in the correct format or is malformed".
    """
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)

    samples, sr = _decode_container(audio_bytes)
    if samples is None:
        # Raw PCM16 at whatever rate the WS session negotiated (== target_sr).
        # Drop a trailing odd byte: frombuffer rejects a buffer that isn't a
        # whole number of samples, and a split frame shouldn't be a 500.
        pcm = audio_bytes[: len(audio_bytes) - (len(audio_bytes) % 2)]
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        sr = target_sr

    samples = _as_float_mono(samples)
    if sr and sr != target_sr and samples.size:
        samples = _resample(samples, sr, target_sr)
    return samples


def _silence_pcm16(samples: int) -> bytes:
    """``samples`` of silence in the raw PCM16 wire format engines expect."""
    return b"\x00\x00" * samples


def _decode_container(audio_bytes: bytes) -> tuple[np.ndarray | None, int]:
    """Parse a self-describing audio file. ``(None, 0)`` if it isn't one."""
    import io

    try:
        import soundfile as sf  # optional dep, kept local
    except ImportError:
        return None, 0
    try:
        samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        return None, 0
    return samples, int(sr)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import librosa  # optional dep, kept local
    except ImportError:
        # Linear interpolation is poorer than librosa's polyphase filter, but a
        # missing optional dep shouldn't turn a decodable request into a 500.
        n = int(round(samples.size * target_sr / orig_sr))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        src = np.arange(samples.size, dtype=np.float32)
        return np.interp(
            np.linspace(0, samples.size - 1, n), src, samples
        ).astype(np.float32)
    return librosa.resample(
        samples, orig_sr=orig_sr, target_sr=target_sr
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Mock STT backend                                                            #
# --------------------------------------------------------------------------- #
@_stt_register
class MockSSEngine(SSEngine):
    """Generates a short synthetic transcript. No deps, no GPU needed."""

    NAME = "mock"
    DEFAULT_MODEL = "mock"
    SAMPLE_RATE = 16000
    MAX_INPUT_SECONDS = 30.0
    LANGUAGES = ["Auto", "en", "fr", "es", "de"]

    def stream_transcribe(
        self, audio_bytes: bytes, *, language: str | None = None,
    ) -> Iterator[str]:
        # Return a deterministic "transcript" whose length scales with audio size.
        seconds = len(audio_bytes) / self.SAMPLE_RATE if self.SAMPLE_RATE else 30
        words = "the quick brown fox jumps over the lazy dog sample transcript"
        chunk_count = max(1, int(seconds * 2))                  # ~2 segments / sec
        word_idx = 0
        for i in range(chunk_count):
            n_words = min(len(words.split()) - word_idx, 5)
            if n_words <= 0:
                n_words = 5
            chunk = " ".join(words.split()[word_idx:word_idx + n_words])
            word_idx = (word_idx + n_words) % len(words.split())
            yield chunk


# --------------------------------------------------------------------------- #
# VoxtralEngine — mistralai/Voxtral-Small-24B-2507 via transformers           #
# --------------------------------------------------------------------------- #
@_stt_register
class VoxtralEngine(SSEngine):
    """Voxtral-Small (8x7B) multimodal speech-to-text via the ``transformers``
    pipeline.

    Voxtral is a multimodal vision-language model, but for audio-only ASR we use
    its **Audio2Text** pipeline which handles encoding + decoding internally.
    """

    NAME = "voxtral"
    DEFAULT_MODEL = "mistralai/Voxtral-Small-24B-2507"
    SAMPLE_RATE = 16000
    MAX_INPUT_SECONDS = 60.0
    LANGUAGES = ["en", "fr", "de", "es", "it", "pt", "ru", "ja", "zh"]

    def __init__(self, settings: Settings, model_id: str | None = None):
        super().__init__(settings, model_id)
        from transformers import pipeline  # heavy import — local

        log.info("Loading Voxtral pipeline for %s on %s ...", self.model_id, settings.device)
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=int(settings.device.split(":")[-1]) if ":0" in settings.device else 0,
            torch_dtype=getattr(__import__("torch"), settings.dtype),
        )

    def warmup(self) -> None:
        try:
            # One second of silence as SAMPLES. Passing bytes here would make
            # transformers read them as an encoded file and invoke ffmpeg.
            list(self.stream_transcribe(_silence_pcm16(self.SAMPLE_RATE)))
            log.info("Warmup complete.")
        except Exception:  # pragma: no cover
            log.exception("Warmup failed (continuing anyway).")

    def stream_transcribe(
        self, audio_bytes: bytes, *, language: str | None = None,
    ) -> Iterator[str]:
        samples = self.decode(audio_bytes)
        if not samples.size:
            return
        # pipeline returns {"text": "..."} for a single chunk; we split into segments.
        result = self._pipeline(
            {"raw": samples, "sampling_rate": self.target_sample_rate},
            generate_kwargs={"language": language} if language else {},
        )
        text = result.get("text", "")
        if text:
            yield text  # Voxtral delivers the whole clip at once; one segment


# --------------------------------------------------------------------------- #
# WhisperEngine — whisper-large-v3 (or any model) via native whisper or pipeline
# --------------------------------------------------------------------------- #
@_stt_register
class WhisperEngine(SSEngine):
    """OpenAI Whisper via the ``transformers`` automatic-speech-recognition
    pipeline.  Supports any whisper model on HuggingFace.

    Fallback: if ``whisper`` (native) is installed, uses it directly for better
    performance; otherwise falls back to the ``pipeline`` approach.
    """

    NAME = "whisper"
    DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
    SAMPLE_RATE = 16000
    MAX_INPUT_SECONDS = 30.0
    LANGUAGES = ["en", "fr", "de", "es", "it", "pt", "ru", "ja", "zh", "auto"]

    def __init__(self, settings: Settings, model_id: str | None = None):
        super().__init__(settings, model_id)
        self._pipeline = None
        self._native = None
        try:
            import whisper as _whisper  # noqa: F401 -- check availability
            self._use_native = True
        except ImportError:
            self._use_native = False

    def _ensure_native(self):
        """Load (once) the native whisper model.

        Native whisper takes a size name ("large-v3"), not a HuggingFace repo
        id, so strip an ``openai/whisper-`` prefix if the configured
        ``STT_MODEL_ID`` is the HF form.
        """
        if self._native is None:
            name = (self.model_id or self.DEFAULT_MODEL).rsplit("/", 1)[-1]
            name = name.removeprefix("whisper-")
            log.info("Loading native whisper model %r on %s ...",
                     name, self.settings.device)
            self._native = __import__("whisper").load_model(
                name, device=self.settings.device,
            )
        return self._native

    def _ensure_pipeline(self):
        if getattr(self, "_pipeline", None) is not None:
            return
        from transformers import pipeline
        log.info("Loading Whisper pipeline for %s on %s ...",
                 self.model_id, self.settings.device)
        dev = int(self.settings.device.split(":")[-1]) if ":0" in self.settings.device else 0
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id or self.DEFAULT_MODEL,
            device=dev,
            torch_dtype=getattr(__import__("torch"), self.settings.dtype),
        )

    def warmup(self) -> None:
        # Best-effort, like every other engine's warmup: this runs inside the
        # lifespan handler, so raising here takes the whole server down at
        # startup rather than failing the one request that needs the model.
        try:
            # One second of silence as SAMPLES. Handing raw bytes to the
            # transformers pipeline makes it read them as an encoded file and
            # shell out to ffmpeg, which is what failed here before.
            list(self.stream_transcribe(_silence_pcm16(self.SAMPLE_RATE)))
            log.info("Warmup complete.")
        except Exception:  # pragma: no cover
            log.exception("Warmup failed (continuing anyway).")

    def stream_transcribe(
        self, audio_bytes: bytes, *, language: str | None = None,
    ) -> Iterator[str]:
        lang = language or "auto"
        # Both paths want mono float32 at 16 kHz, whether the caller sent a
        # file (HTTP) or raw PCM16 frames (WebSocket).
        samples = self.decode(audio_bytes)
        if not samples.size:
            return

        # ---- native whisper path (faster, less VRAM) ------------------------ #
        if self._use_native:
            result = self._ensure_native().transcribe(
                samples, language=None if lang == "auto" else lang,
            )
            for seg in result.get("segments", []):
                yield seg["text"]

        # ---- transformers pipeline fallback -------------------------------- #
        else:
            self._ensure_pipeline()
            gen_kwargs = {} if lang == "auto" else {"language": lang}
            result = self._pipeline(
                {"raw": samples, "sampling_rate": self.target_sample_rate},
                generate_kwargs=gen_kwargs,
            )
            text = result.get("text", "")
            if text:
                yield text
