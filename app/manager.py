"""Multi-model manager.

Holds one :class:`Synthesizer` (engine + concurrency controls) per model id,
building each lazily on first use. Model building is blocking (the real backend
loads a multi-gigabyte model onto the GPU), so it runs in a thread executor and
is guarded so two concurrent requests for the same new model load it once.
"""
from __future__ import annotations

import asyncio
import logging

from .config import Settings
from .engine import build_engine
from .streaming import Synthesizer

log = logging.getLogger("tts.manager")

# Generic aliases clients may send (e.g. the OpenAI SDK) -> resolved to default.
_GENERIC_ALIASES = {
    "", "default", "auto",
    "tts-1", "tts-1-hd",
    "qwen3-tts", "qwen",
    "kokoro", "kokoro-82m",
}


class UnknownModelError(KeyError):
    """Raised when a requested model is not in the allow-list."""


class EngineManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.default_model = settings.model_id
        self._synths: dict[str, Synthesizer] = {}
        self._build_lock = asyncio.Lock()

    @property
    def available(self) -> list[str]:
        return self.settings.model_list

    def resolve(self, model: str | None) -> str:
        """Map a requested model name to a concrete model id."""
        if model is None or model.strip().lower() in _GENERIC_ALIASES:
            return self.default_model
        return model.strip()

    async def get(self, model: str | None = None) -> Synthesizer:
        """Return the Synthesizer for ``model``, building it if needed."""
        model_id = self.resolve(model)

        # For real backends, restrict to the allow-list so a request can't
        # trigger an arbitrary multi-gigabyte download. The mock backend has no
        # such cost, so it happily fabricates any requested name.
        if (self.settings.backend != "mock"
                and model_id not in set(self.available)):
            raise UnknownModelError(model_id)

        if model_id in self._synths:
            return self._synths[model_id]

        async with self._build_lock:
            if model_id not in self._synths:  # re-check under lock
                loop = asyncio.get_running_loop()
                engine = await loop.run_in_executor(
                    None, build_engine, self.settings, model_id
                )
                self._synths[model_id] = Synthesizer(engine, self.settings)
                log.info("Model ready: %s", model_id)
        return self._synths[model_id]

    async def preload_default(self) -> None:
        synth = await self.get(self.default_model)
        synth.engine.warmup()
