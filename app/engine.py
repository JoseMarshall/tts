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
* ``DiaEngine``    — Dia-1.6B dialogue model via the ``dia`` package.
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
            f"Unknown backend {name!r}. Available: {available_backends()}"
        )


def engine_default_model(name: str) -> str:
    """The canonical model id for a backend (used when a client selects by
    backend name, e.g. model='kokoro')."""
    return engine_class(name).DEFAULT_MODEL or name


class TTSEngine(abc.ABC):
    # ---- backend metadata (override in subclasses) ------------------------ #
    NAME: str = ""
    DEFAULT_MODEL: str = ""                  # canonical model id for this backend
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
    DEFAULT_MODEL = "mock"

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
    DEFAULT_MODEL = "hexgrad/Kokoro-82M"
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
            if audio is None:      # segment produced no audio (e.g. a pause)
                continue
            audio = _as_float_mono(audio)
            for s in range(0, len(audio), step):
                frame = audio[s:s + step]
                if frame.size:
                    yield frame


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
# SST (Speech-to-text) engines — parallel hierarchy                           #
# --------------------------------------------------------------------------- #

_SST_REGISTRY: dict[str, type["SSEngine"]] = {}


def _sst_register(cls: type["SSEngine"]) -> type["SSEngine"]:
    """Class decorator: register an SST engine under its ``NAME``."""
    if not getattr(cls, "NAME", None):
        raise ValueError(f"{cls.__name__} must define a NAME")
    _SST_REGISTRY[cls.NAME] = cls
    return cls


def sst_available_backends() -> list[str]:
    return sorted(_SST_REGISTRY)


def sst_engine_class(name: str) -> type["SSEngine"]:
    try:
        return _SST_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown SST backend {name!r}. Available: {sst_available_backends()}"
        )


def sst_engine_default_model(name: str) -> str:
    return sst_engine_class(name).DEFAULT_MODEL or name


class SSEngine(abc.ABC):
    """Abstract base for all speech-to-text engines.

    An SST engine takes audio bytes and produces text.  Streaming engines can
    yield partial segments while a non-streaming one waits for the complete
    result before returning.

    Adding a new SST backend mirrors the TTS pattern — register, define metadata,
    implement ``__init__`` (heavy imports here) and ``stream_transcribe()``. No
    changes are needed in routing, framing or the manager.

    Backends bundled here:
    * ``MockEngine``     -- dependency-free placeholder that echoes back the first
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
            getattr(settings, "sst_model_id", "") or sst_engine_default_model(self.NAME)
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

    def _language(self, req_language: str | None) -> str | None:
        """Resolve a language hint, falling through settings → backend default."""
        # Accept either the TTS-style or SST-style field name for compatibility.
        return (req_language
                or getattr(self.settings, "default_language", None)
                or getattr(self.settings, "sst_default_language", None))

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
# Mock SST backend                                                            #
# --------------------------------------------------------------------------- #
@_sst_register
class MockEngine(SSEngine):
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
@_sst_register
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
            list(self.stream_transcribe(b"\x00" * self.SAMPLE_RATE))
            log.info("Warmup complete.")
        except Exception:  # pragma: no cover
            log.exception("Warmup failed (continuing anyway).")

    def stream_transcribe(
        self, audio_bytes: bytes, *, language: str | None = None,
    ) -> Iterator[str]:
        # pipeline returns {"text": "..."} for a single chunk; we split into segments.
        result = self._pipeline(
            np.frombuffer(audio_bytes, dtype=np.float32)[:self.SAMPLE_RATE * self.MAX_INPUT_SECONDS],
            generate_kwargs={"language": language} if language else None,
        )
        text = result.get("text", "")
        if text:
            yield text  # Voxtral delivers the whole clip at once; one segment


# --------------------------------------------------------------------------- #
# WhisperEngine — whisper-large-v3 (or any model) via native whisper or pipeline
# --------------------------------------------------------------------------- #
@_sst_register
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
        try:
            import whisper as _whisper  # noqa: F401 -- check availability
            self._use_native = True
        except ImportError:
            self._use_native = False

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
        if self._use_native:
            whisper_model = __import__("whisper").load_model(
                self.model_id or self.DEFAULT_MODEL,
                device=self.settings.device,
            )
            whisper_model.transcribe(b"\x00" * 16000)
        else:
            self._ensure_pipeline()
            self._pipeline(b"\x00" * 16000)
        log.info("Warmup complete.")

    def stream_transcribe(
        self, audio_bytes: bytes, *, language: str | None = None,
    ) -> Iterator[str]:
        lang = language or "auto"

        # ---- native whisper path (faster, less VRAM) ------------------------ #
        if self._use_native:
            import soundfile as sf  # for loading audio
            model = __import__("whisper").load_model(
                self.model_id or self.DEFAULT_MODEL,
                device=self.settings.device,
            )
            # whisper expects mono float32 PCM at 16kHz
            try:
                samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", samplerate=16000)
            except Exception:
                samples = np.frombuffer(audio_bytes, dtype=np.float32)[:16000 * self.MAX_INPUT_SECONDS]
            result = model.transcribe(
                samples if sr == 16000 else librosa.resample(samples, orig_sr=sr, target_sr=16000),
                language=lang if lang != "auto" else None,
            )
            for seg in result.get("segments", []):
                yield seg["text"]

        # ---- transformers pipeline fallback -------------------------------- #
        else:
            self._ensure_pipeline()
            samples = np.frombuffer(audio_bytes, dtype=np.float32)[:self.SAMPLE_RATE * self.MAX_INPUT_SECONDS]
            gen_kwargs = {"language": lang} if lang != "auto" else None
            result = self._pipeline(samples, generate_kwargs=gen_kwargs or {})
            text = result.get("text", "")
            if text:
                yield text
