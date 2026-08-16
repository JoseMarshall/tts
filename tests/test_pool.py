"""Tests for concurrent generation via the in-process engine pool.

The property worth protecting is that ``TTS_ENGINE_REPLICAS=1`` — the default —
is byte-for-byte the old ``Semaphore(1)`` behaviour, and that anything above it
actually runs concurrently rather than merely claiming to. Both are proved with
a barrier: two generations that must meet inside ``stream()`` can only do so if
they really are running at the same time.
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engine import DiaEngine, KokoroEngine, MockEngine, QwenEngine
from app.main import app
from app.manager import EngineManager
from app.schemas import TTSRequest
from app.streaming import EnginePool, Synthesizer


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class _BarrierEngine(MockEngine):
    """Its ``stream()`` cannot finish alone — two must meet inside it.

    A pool of one therefore times out, and a pool of two completes. That is a
    direct test of concurrency rather than of a counter.
    """

    SUPPORTS_REPLICAS = True
    barrier: threading.Barrier | None = None

    def stream(self, req):
        assert self.barrier is not None
        try:
            self.barrier.wait(timeout=1.5)
        except threading.BrokenBarrierError:
            raise RuntimeError("generations did not overlap")
        yield np.zeros(128, dtype=np.float32)


def _settings(**kw):
    return Settings(backend="mock", **kw)


# --------------------------------------------------------------------------- #
# EnginePool
# --------------------------------------------------------------------------- #
def test_pool_of_one_is_the_old_behaviour():
    settings = _settings()

    async def scenario():
        engine = _BarrierEngine(settings)
        engine.barrier = threading.Barrier(2)
        synth = Synthesizer(EnginePool(engine), settings)
        assert synth.pool.size == 1
        req = TTSRequest(text="hi")
        return await asyncio.gather(
            synth.synthesize_bytes(req), synth.synthesize_bytes(req),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    # Serialised: neither generation ever meets the other at the barrier.
    assert any(isinstance(r, Exception) for r in results)


def test_pool_of_two_generates_concurrently():
    settings = _settings()
    barrier = threading.Barrier(2)

    def build():
        eng = _BarrierEngine(settings)
        eng.barrier = barrier
        return eng

    async def scenario():
        primary = build()
        pool = EnginePool(primary, build=build, size=2)
        await pool.fill()
        assert pool.size == 2
        synth = Synthesizer(pool, settings)
        req = TTSRequest(text="hi")
        return await asyncio.gather(
            synth.synthesize_bytes(req), synth.synthesize_bytes(req)
        )

    a, b = asyncio.run(scenario())
    assert len(a) > 0 and len(b) > 0     # both completed; the barrier was met


def test_pool_returns_engine_after_an_error():
    """A generation that raises must not leak its replica out of the pool."""
    settings = _settings()

    class _Boom(MockEngine):
        def stream(self, req):
            raise ValueError("nope")
            yield  # pragma: no cover

    async def scenario():
        pool = EnginePool(_Boom(settings))
        synth = Synthesizer(pool, settings)
        for _ in range(3):
            with pytest.raises(ValueError):
                await synth.synthesize_bytes(TTSRequest(text="hi"))
        # Still exactly one engine available, so the next call can proceed.
        async with pool.acquire() as engine:
            assert engine is pool.primary

    asyncio.run(scenario())


def test_pool_fill_warms_each_replica():
    settings = _settings()
    warmed: list[int] = []

    class _CountingEngine(MockEngine):
        SUPPORTS_REPLICAS = True

        def warmup(self):
            warmed.append(1)

    async def scenario():
        pool = EnginePool(
            _CountingEngine(settings), build=lambda: _CountingEngine(settings), size=3
        )
        await pool.fill()
        return pool

    pool = asyncio.run(scenario())
    assert pool.size == 3
    # The primary is warmed by the manager; fill() warms the two it builds,
    # so no request ever lands on a cold instance.
    assert len(warmed) == 2


def test_pool_survives_a_replica_that_will_not_build():
    settings = _settings()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            return MockEngine(settings)
        raise RuntimeError("out of VRAM")

    async def scenario():
        pool = EnginePool(MockEngine(settings), build=flaky, size=4)
        await pool.fill()
        # Degrades to the replicas it managed, and stays usable.
        assert pool.size == 2
        assert pool.target == 2
        async with pool.acquire() as engine:
            assert engine is not None
        return pool

    asyncio.run(scenario())


def test_synthesizer_still_accepts_a_bare_engine():
    """Back-compat: the manager passes a pool, everything else passes an engine."""
    settings = _settings()
    engine = MockEngine(settings)
    synth = Synthesizer(engine, settings)
    assert synth.engine is engine
    assert synth.pool.primary is engine
    assert synth.pool.size == 1
    assert synth.sample_rate == engine.sample_rate
    assert synth.model_id == engine.model_id


# --------------------------------------------------------------------------- #
# Capability flag and manager wiring
# --------------------------------------------------------------------------- #
def test_supports_replicas_is_opt_in():
    assert MockEngine.SUPPORTS_REPLICAS is True
    for cls in (QwenEngine, KokoroEngine, DiaEngine):
        assert cls.SUPPORTS_REPLICAS is False, (
            f"{cls.__name__} must verify it has no shared global state "
            "(espeak-ng, module caches) before declaring replica support"
        )
    assert MockEngine.capabilities()["supports_replicas"] is True
    assert KokoroEngine.capabilities()["supports_replicas"] is False


def test_manager_builds_the_configured_replicas():
    async def scenario():
        manager = EngineManager(_settings(engine_replicas=3))
        synth = await manager.get()
        # get() returns as soon as one replica is usable; the rest fill behind.
        assert synth.pool.target == 3
        for _ in range(50):
            if synth.pool.size == 3:
                break
            await asyncio.sleep(0.01)
        return synth.pool

    pool = asyncio.run(scenario())
    assert pool.size == 3


def test_manager_refuses_replicas_a_backend_has_not_declared(monkeypatch, caplog):
    monkeypatch.setattr(MockEngine, "SUPPORTS_REPLICAS", False)

    async def scenario():
        manager = EngineManager(_settings(engine_replicas=4))
        return await manager.get()

    with caplog.at_level("WARNING"):
        synth = asyncio.run(scenario())

    assert synth.pool.target == 1        # not 4: wrong output beats a crash
    assert "SUPPORTS_REPLICAS" in caplog.text


def test_manager_default_is_one_replica():
    async def scenario():
        manager = EngineManager(_settings())
        return await manager.get()

    synth = asyncio.run(scenario())
    assert synth.pool.size == 1 and synth.pool.target == 1


def test_health_reports_replicas(client):
    body = client.get("/health").json()
    assert "replicas" in body
    # The default model is preloaded at startup, so it is loaded and counted.
    assert body["replicas"], "expected at least the preloaded default model"
    assert all(isinstance(n, int) and n >= 1 for n in body["replicas"].values())
    assert set(body["replicas"]) == set(body["loaded"])


def test_voices_advertises_supports_replicas(client):
    assert client.get("/v1/voices").json()["supports_replicas"] is True
