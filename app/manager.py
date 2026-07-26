"""Multi-backend, multi-model manager.

A single server can host several backends at once (Qwen, Kokoro, Dia, …) and the
client picks per request via the ``model`` field — either by **backend name**
(``model="kokoro"`` → that backend's default model) or by a **model id**
(``model="hexgrad/Kokoro-82M"``). Each distinct model is loaded lazily on first
use and cached, so advertising several backends costs nothing until one is used.

What a client may select is operator-controlled:

* the default model (``TTS_BACKEND`` + ``TTS_MODEL_ID``), always available;
* any backend listed in ``TTS_BACKENDS`` (selectable by name → its default model);
* any specific model listed in ``TTS_MODELS`` (``backend:model_id`` entries).

Anything else returns 400 — so a request can't trigger an arbitrary,
multi-gigabyte download. (The ``mock`` backend is permissive, for tests.)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .config import Settings
from .engine import (
    available_backends,
    engine_class,
    engine_default_model,
)
from .streaming import Synthesizer

log = logging.getLogger("tts.manager")

# Truly generic aliases that always mean "the server default model".
_DEFAULT_ALIASES = {"", "default", "auto", "tts-1", "tts-1-hd"}


class UnknownModelError(KeyError):
    """Raised when a requested model/backend is not selectable."""


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    model_id: str

    @property
    def key(self) -> str:
        return f"{self.backend}:{self.model_id}"


def _parse_entry(entry: str, default_backend: str) -> ModelSpec:
    """Parse a TTS_MODELS entry: 'backend:model_id', 'backend', or 'model_id'."""
    backends = available_backends()
    if ":" in entry:
        head, tail = entry.split(":", 1)
        if head in backends:
            return ModelSpec(head, tail or engine_default_model(head))
    if entry in backends:  # bare backend name -> its default model
        return ModelSpec(entry, engine_default_model(entry))
    return ModelSpec(default_backend, entry)  # bare model id, default backend


class EngineManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._synths: dict[str, Synthesizer] = {}  # spec.key -> Synthesizer
        self._build_lock = asyncio.Lock()

        # Default model.
        default_backend = settings.backend
        default_model_id = settings.model_id or engine_default_model(default_backend)
        self.default_spec = ModelSpec(default_backend, default_model_id)

        # Catalog of explicitly selectable model ids -> spec.
        self._catalog: dict[str, ModelSpec] = {
            self.default_spec.model_id: self.default_spec
        }
        # Backends the client may select by name -> that backend's default spec.
        self._enabled_backends: set[str] = {default_backend}

        for name in settings.extra_backends:
            engine_class(name)  # validate
            self._enabled_backends.add(name)
            spec = ModelSpec(name, engine_default_model(name))
            self._catalog.setdefault(spec.model_id, spec)

        for entry in settings.model_entries:
            spec = _parse_entry(entry, default_backend)
            engine_class(spec.backend)  # validate
            self._enabled_backends.add(spec.backend)
            self._catalog[spec.model_id] = spec

    # ---- introspection ---------------------------------------------------- #
    @property
    def default_model(self) -> str:
        return self.default_spec.model_id

    @property
    def enabled_backends(self) -> list[str]:
        return sorted(self._enabled_backends)

    @property
    def catalog(self) -> list[ModelSpec]:
        # Deduplicate by key, stable order.
        seen: dict[str, ModelSpec] = {}
        for spec in self._catalog.values():
            seen.setdefault(spec.key, spec)
        return list(seen.values())

    @property
    def available(self) -> list[str]:
        """Selectable identifiers: model ids plus enabled backend-name aliases."""
        ids = [s.model_id for s in self.catalog]
        return sorted(set(ids) | set(self._enabled_backends))

    # ---- resolution ------------------------------------------------------- #
    def resolve(self, model: str | None) -> ModelSpec:
        if model is None or model.strip().lower() in _DEFAULT_ALIASES:
            return self.default_spec
        m = model.strip()
        low = m.lower()

        # 1) a backend name the operator enabled -> its default model
        if low in self._enabled_backends and low in set(available_backends()):
            return ModelSpec(low, engine_default_model(low))

        # 2) explicit "backend:model_id" for an enabled backend
        if ":" in m:
            head, tail = m.split(":", 1)
            if head in self._enabled_backends:
                spec = ModelSpec(head, tail)
                if spec.model_id in self._catalog or head == "mock":
                    return spec

        # 3) a catalog model id
        if m in self._catalog:
            return self._catalog[m]

        # 4) mock default backend is permissive (used by tests/dev)
        if self.settings.backend == "mock":
            return ModelSpec("mock", m)

        raise UnknownModelError(model)

    async def get(self, model: str | None = None) -> Synthesizer:
        """Return the Synthesizer for the requested model, building it lazily."""
        spec = self.resolve(model)
        if spec.key in self._synths:
            return self._synths[spec.key]

        async with self._build_lock:
            if spec.key not in self._synths:  # re-check under lock
                loop = asyncio.get_running_loop()
                engine = await loop.run_in_executor(
                    None, self._build, spec
                )
                self._synths[spec.key] = Synthesizer(engine, self.settings)
                log.info("Model ready: %s", spec.key)
        return self._synths[spec.key]

    def _build(self, spec: ModelSpec):
        # Build the right engine class for this spec's backend.
        return engine_class(spec.backend)(self.settings, spec.model_id)

    async def preload_default(self) -> None:
        synth = await self.get(self.default_spec.model_id)
        synth.engine.warmup()
