"""Request/response models for the TTS API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ResponseFormat = Literal["wav", "pcm"]
VoiceMode = Literal["custom_voice", "voice_design", "voice_clone"]


class TTSRequest(BaseModel):
    """Native synthesis request.

    Fields are intentionally generic; each backend interprets what it supports
    and raises for what it doesn't (e.g. Kokoro rejects voice cloning). Query
    ``GET /v1/voices`` for the active backend's speakers and languages.

    The synthesis mode is inferred unless ``mode`` is set explicitly:
      * ``ref_audio`` present -> ``voice_clone``
      * otherwise             -> ``custom_voice`` (using ``speaker``, and
        ``instruct`` as an optional style modifier where supported)

    ``voice_design`` (a voice from an ``instruct`` description alone) is never
    inferred — request it explicitly with ``mode="voice_design"`` on a backend
    that provides it.
    """

    text: str = Field(..., min_length=1, description="Text to synthesise.")
    language: str = Field(
        "Auto", description="Language name or code; backend-specific ('Auto' ok)."
    )
    model: Optional[str] = Field(
        None,
        description="Which model/backend to use. Either a backend name "
                    "(e.g. 'qwen', 'kokoro', 'dia') or a model id "
                    "(e.g. 'hexgrad/Kokoro-82M'). Omit for the server default. "
                    "Must be selectable — see GET /v1/models.",
    )

    # Explicit mode override; inferred when omitted (see class docstring).
    mode: Optional[VoiceMode] = Field(
        None,
        description="Force a synthesis mode. Omit to infer (ref_audio -> "
                    "voice_clone, else custom_voice).",
    )
    speaker: Optional[str] = Field(
        None, description="Preset voice/speaker name (backend default if omitted)."
    )
    speed: float = Field(
        1.0, gt=0, le=4.0,
        description="Speech-rate multiplier for backends that support it (Kokoro).",
    )
    # Style instruction: a modifier for custom_voice, or the description that
    # drives voice_design (backend-dependent).
    instruct: Optional[str] = Field(
        None, description="Natural-language voice/style instruction."
    )
    # Voice cloning
    ref_audio: Optional[str] = Field(
        None,
        description="Reference audio for cloning: file path, URL, or base64.",
    )
    ref_text: Optional[str] = Field(
        None, description="Transcript of the reference audio (for cloning)."
    )

    response_format: ResponseFormat = "wav"

    def resolve_mode(self) -> str:
        if self.mode:
            return self.mode
        if self.ref_audio:
            return "voice_clone"
        # `instruct` modifies custom_voice; it does not imply voice_design.
        return "custom_voice"


class OpenAISpeechRequest(BaseModel):
    """Subset of the OpenAI ``/v1/audio/speech`` schema, mapped onto our engines."""

    model: str = "tts-1"
    input: str = Field(..., min_length=1)
    voice: str = ""                       # empty -> backend default
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = 1.0
    # Non-standard extras (honoured if supplied):
    language: str = "Auto"
    instructions: Optional[str] = None
