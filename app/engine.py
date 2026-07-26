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
model-agnostic and need no changes.

Backends bundled here:
* ``MockEngine``   — dependency-free tone generator (no GPU / no downloads).
* ``QwenEngine``   — Qwen3-TTS via the ``qwen-tts`` package.
* ``KokoroEngine`` — Kokoro-82M via the ``kokoro`` package.
"""
from __future__ import annotations

import abc
import logging
from typing import Iterator

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
            f"Unknown TTS_BACKEND {name!r}. Available: {available_backends()}"
        )


class TTSEngine(abc.ABC):
    # ---- backend metadata (override in subclasses) ------------------------ #
    NAME: str = ""
    SAMPLE_RATE: int | None = None          # None -> use settings.sample_rate
    SPEAKERS: list[str] = []
    LANGUAGES: list[str] = []
    DEFAULT_SPEAKER: str = ""
    DEFAULT_LANGUAGE: str = ""

    def __init__(self, settings: Settings, model_id: str | None = None):
        self.settings = settings
        self.model_id = model_id or settings.model_id
        self.sample_rate = self.SAMPLE_RATE or settings.sample_rate

    @abc.abstractmethod
    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
        """Yield successive float32 mono audio chunks for ``req``."""

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
        self.sample_rate = int(sr) or self.sample_rate
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
    SAMPLE_RATE = 24000
    DEFAULT_SPEAKER = "af_heart"
    DEFAULT_LANGUAGE = "a"
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

        # Kokoro yields per text segment (naturally streaming). Re-slice each
        # segment into fixed frames for smooth, consistent delivery downstream.
        for result in pipeline(req.text, voice=voice, speed=speed):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            audio = _as_float_mono(audio)
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
