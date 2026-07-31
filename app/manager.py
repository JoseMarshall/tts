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

from .config import Settings, get_settings, _SSTSettings
from .engine import (
    SSEngine,
    available_backends,
    engine_class,
    engine_default_model,
    sst_available_backends,
    sst_engine_class,
    sst_engine_default_model,
)
from .streaming import Transcriber  # forward-ref to avoid circular issues; imported at runtime via module-level access
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


# --------------------------------------------------------------------------- #
# SST (Speech-to-text) multi-model manager                                    #
# --------------------------------------------------------------------------- #
"""Parallel to ``EngineManager`` but for the SST side of the server.

The server hosts BOTH text-to-speech and speech-to-text: a single process serves
TTS ``/v1/tts`` endpoints and SST ``/v1/sst`` endpoints at the same time, each
with its own engine catalog, lazy-loader and streaming bridge.

Client selection mirrors TTS exactly:
* default model  →  ``SST_BACKEND`` + ``SST_MODEL_ID``
* extra backends →  ``SST_BACKENDS`` (selectable by backend name)
* specific models →  ``SST_MODELS`` (``backend:model_id`` entries)
"""

_DEFAULT_ALIASES = {"", "default", "auto"}


class SSTUnknownModelError(KeyError):
    """Raised when a requested SST model/backend is not selectable."""


@dataclass(frozen=True)
class SSTModelSpec:
    backend: str
    model_id: str

    @property
    def key(self) -> str:
        return f"{self.backend}:{self.model_id}"


def _parse_sst_entry(entry: str, default_backend: str) -> SSTModelSpec:
    """Parse an SST_MODELS entry: ``'backend:model_id'``, ``'backend'``, or ``'model_id'``."""
    backends = sst_available_backends()
    if ":" in entry:
        head, tail = entry.split(":", 1)
        if head in backends:
            return SSTModelSpec(head, tail or sst_engine_default_model(head))
    if entry in backends:
        return SSTModelSpec(entry, sst_engine_default_model(entry))
    return SSTModelSpec(default_backend, entry)


class SSTManager:
    """Manages multiple SST backend models with lazy loading."""

    def __init__(self, sst_settings: _SSTSettings):
        self._settings = sst_settings
        self._transcribers: dict[str, "Transcriber"] = {}  # spec.key -> Transcriber
        self._build_lock = asyncio.Lock()

        # Default model.
        default_backend = sst_settings.backend or "mock"
        default_model_id = sst_settings.model_id or sst_engine_default_model(default_backend)
        self.default_spec = SSTModelSpec(default_backend, default_model_id)

        # Catalog of explicitly selectable model ids -> spec.
        self._catalog: dict[str, SSTModelSpec] = {
            self.default_spec.model_id: self.default_spec
        }
        self._enabled_backends: set[str] = {default_backend}

        for name in sst_settings.extra_backends:
            sst_engine_class(name)  # validate
            self._enabled_backends.add(name)
            spec = SSTModelSpec(name, sst_engine_default_model(name))
            self._catalog.setdefault(spec.model_id, spec)

        for entry in sst_settings.model_entries:
            spec = _parse_sst_entry(entry, default_backend)
            sst_engine_class(spec.backend)  # validate
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
    def catalog(self) -> list[SSTModelSpec]:
        seen: dict[str, SSTModelSpec] = {}
        for spec in self._catalog.values():
            seen.setdefault(spec.key, spec)
        return list(seen.values())

    @property
    def available(self) -> list[str]:
        ids = [s.model_id for s in self.catalog]
        return sorted(set(ids) | set(self._enabled_backends))

    # ---- resolution ------------------------------------------------------- #
    def resolve(self, model: str | None) -> SSTModelSpec:
        if model is None or model.strip().lower() in _DEFAULT_ALIASES:
            return self.default_spec
        m = model.strip().lower()

        if m in self._enabled_backends and m in set(sst_available_backends()):
            return SSTModelSpec(m, sst_engine_default_model(m))

        full = model.strip()  # preserve casing for model-id lookup
        if ":" in full:
            head, tail = full.split(":", 1)
            if head.lower() in self._enabled_backends:
                spec = SSTModelSpec(head, tail)
                if spec.model_id in self._catalog or head == "mock":
                    return spec

        if full in self._catalog:
            return self._catalog[full]

        if self._settings.backend == "mock":
            return SSTModelSpec("mock", full)

        raise SSTUnknownModelError(model)

    async def get(self, model: str | None = None) -> object:
        """Return a Transcriber for the requested model, building it lazily.

        A ``Transcriber`` wraps one SSEngine with concurrency controls similar
        to ``Synthesizer`` but adapted for **audio-in → text-out** (binary input,
        JSON text output) rather than text-in → audio chunks.
        """
        spec = self.resolve(model)
        if spec.key in self._transcribers:
            return self._transcribers[spec.key]

        async with self._build_lock:
            if spec.key not in self._transcribers:
                loop = asyncio.get_running_loop()
                engine_cls = sst_engine_class(spec.backend)
                engine = await loop.run_in_executor(
                    None, self._build_engine, engine_cls, spec.model_id)
                self._transcribers[spec.key] = Transcriber(engine, self._settings)
                log.info("SST model ready: %s", spec.key)
        return self._transcribers[spec.key]

    def _build_engine(self, engine_cls: type["SSEngine"], model_id: str):
        return engine_cls(self._settings, model_id)

    async def preload_default(self) -> None:
        transcriber = await self.get(self.default_spec.model_id)
        # type ignore because `engine` is always set on instance
        transcriber.engine.warmup()  # type: ignore[attr-defined]
