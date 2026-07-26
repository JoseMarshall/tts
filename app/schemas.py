"""Request/response models for the TTS API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Advertised in /v1/voices and validated loosely (the model may support more).
SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

ResponseFormat = Literal["wav", "pcm"]


class TTSRequest(BaseModel):
    """Native synthesis request.

    Exactly one voice source is used, chosen in this order of precedence:
      1. ``ref_audio`` present  -> voice cloning
      2. ``instruct`` present   -> voice design
      3. otherwise              -> custom voice from ``speaker``
    """

    text: str = Field(..., min_length=1, description="Text to synthesise.")
    language: str = Field("Auto", description="Language hint or 'Auto'.")
    model: Optional[str] = Field(
        None,
        description="Model id to use. Omit for the server default; must be one "
                    "of the allow-listed models (see GET /v1/models).",
    )

    # Custom voice
    speaker: Optional[str] = Field(
        None, description="Preset speaker name (custom-voice mode)."
    )
    # Voice design
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

    @field_validator("language")
    @classmethod
    def _check_language(cls, v: str) -> str:
        if not v:
            return "Auto"
        match = {lang.lower(): lang for lang in LANGUAGES}.get(v.lower())
        if match is None:
            raise ValueError(
                f"Unsupported language {v!r}. Supported: {', '.join(LANGUAGES)}"
            )
        return match

    def mode(self) -> str:
        if self.ref_audio:
            return "voice_clone"
        if self.instruct:
            return "voice_design"
        return "custom_voice"


class OpenAISpeechRequest(BaseModel):
    """Subset of the OpenAI ``/v1/audio/speech`` schema, adapted to Qwen3-TTS."""

    model: str = "qwen3-tts"
    input: str = Field(..., min_length=1)
    voice: str = "Vivian"
    # OpenAI uses these format names; we map them onto what we can emit.
    response_format: Literal["wav", "pcm"] = "wav"
    # Non-standard extras (honoured if supplied):
    language: str = "Auto"
    instructions: Optional[str] = None
