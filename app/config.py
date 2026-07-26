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
    # Any registered engine name: "mock" (placeholder tone, no deps),
    # "qwen" (Qwen3-TTS), "kokoro" (Kokoro-82M), ... Validated at startup
    # against the engine registry (see app/engine.py).
    backend: str = "mock"

    # ---- Model loading -----------------------------------------------------
    # ``model_id`` is the default model. ``models`` is a comma-separated
    # allowlist of additional model ids a request may select (the default is
    # always allowed). Requesting a model outside this set returns 400.
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    models: str = ""
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "flash_attention_2"

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
    max_queue: int = 32

    # ---- Server ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: str = ""  # comma-separated; empty disables auth

    @property
    def allowed_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def model_list(self) -> list[str]:
        """The default model followed by any extra allow-listed models."""
        ids = [m.strip() for m in self.models.split(",") if m.strip()]
        result = [self.model_id]
        for m in ids:
            if m not in result:
                result.append(m)
        return result


@lru_cache
def get_settings() -> Settings:
    return Settings()
