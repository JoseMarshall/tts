"""Application configuration.

Settings are read from environment variables (prefixed with ``TTS_``) or a
local ``.env`` file. See ``.env.example`` for the full list.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Backend selection -------------------------------------------------
    # ``backend`` is the DEFAULT engine used when a request doesn't select one.
    # ``backends`` additionally enables other engines so a CLIENT can pick them
    # per request via the ``model`` field (by backend name, e.g. model="kokoro",
    # or by a model id). Comma-separated; the default backend is always enabled.
    # Any registered engine name: "mock", "qwen", "kokoro", "dia", …
    #   e.g. TTS_BACKEND=qwen  TTS_BACKENDS=kokoro,dia
    backend: str = "mock"
    backends: str = ""

    # ---- Model loading -----------------------------------------------------
    # ``model_id`` is the default model for the default backend. Empty -> the
    # default backend's own default model (e.g. Kokoro -> hexgrad/Kokoro-82M).
    # ``models`` is a comma-separated allow-list of ADDITIONAL selectable models,
    # each "backend:model_id" (or "model_id" for the default backend), e.g.
    #   TTS_MODELS=qwen:Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice,kokoro:hexgrad/Kokoro-82M
    # Requesting a model outside the catalog returns 400.
    model_id: str = ""
    models: str = ""
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "sdpa"

    # ---- Audio -------------------------------------------------------------
    sample_rate: int = 24000
    # Size of each streamed audio frame, in samples. Smaller = lower latency,
    # more overhead. ~50 ms at 24 kHz is a good default.
    stream_chunk_samples: int = 1200

    # ---- Generation defaults ----------------------------------------------
    # Empty means "let the active backend pick its own default" (e.g. Qwen ->
    # Vivian/Auto, Kokoro -> af_heart/American English). Set to pin a default.
    default_language: str = ""
    default_speaker: str = ""

    # ---- Concurrency -------------------------------------------------------
    # A single model instance is not safe for concurrent generation, so calls
    # are serialised. This bounds how many requests may queue before we reject.
    # ---- Server ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: str = ""  # comma-separated; empty disables auth

    @property
    def allowed_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def extra_backends(self) -> list[str]:
        return [b.strip() for b in self.backends.split(",") if b.strip()]

    @property
    def model_entries(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]


# --------------------------------------------------------------------------- #
# SST (Speech-to-text) configuration — uses ``SST_`` prefix                  #
# --------------------------------------------------------------------------- #
class _SSTSettings(BaseSettings):
    """Reads environment variables with the ``SST_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="SST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Backend / catalog -------------------------------------------------
    backend: str = "mock"       # default SST engine (when request omits one)
    backends: str = ""          # comma-sep extra backends a client may select
    model_id: str = ""          # specific model id within the default backend
    models: str = ""            # additional selectable ``backend:model_id`` entries
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # Audio
    sample_rate: int = 16000               # target sample rate for input audio
    stream_chunk_samples: int = 1600       # ~100 ms at 16 kHz per chunk
    max_input_seconds: float = 30.0        # reject longer segments to guard GPU

    # ---- Concurrency / Server ----------------------------------------------
    max_queue: int = 32
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: str = ""

    @property
    def allowed_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def extra_backends(self) -> list[str]:
        return [b.strip() for b in self.backends.split(",") if b.strip()]

    @property
    def model_entries(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_sst_settings() -> _SSTSettings:
    return _SSTSettings()
