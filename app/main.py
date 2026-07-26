"""FastAPI application: native, OpenAI-compatible, and WebSocket TTS endpoints."""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse

from .audio import CONTENT_TYPES, pcm_to_wav, wav_header
from .config import Settings, get_settings
from .engine import available_backends, engine_class
from .manager import EngineManager, UnknownModelError
from .schemas import OpenAISpeechRequest, TTSRequest
from .streaming import Synthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.backend not in available_backends():
        raise RuntimeError(
            f"Unknown TTS_BACKEND {settings.backend!r}. "
            f"Available: {available_backends()}"
        )
    log.info("Starting TTS server (backend=%s, models=%s)",
             settings.backend, settings.model_list)
    app.state.manager = EngineManager(settings)
    await app.state.manager.preload_default()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Multi-backend TTS Server",
    version="1.2.0",
    description="Streaming text-to-speech server (Qwen3-TTS, Kokoro-82M, …).",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers / dependencies
# --------------------------------------------------------------------------- #
def get_manager(request: Request) -> EngineManager:
    return request.app.state.manager


async def resolve_synth(manager: EngineManager, model: str | None) -> Synthesizer:
    try:
        return await manager.get(model)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {exc.args[0]!r}. Available: {manager.available}",
        )


def _valid_token(authorization: str | None, settings: Settings) -> bool:
    keys = settings.allowed_keys
    if not keys:
        return True
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    return token in keys


async def require_auth(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not _valid_token(authorization, settings):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _stream_headers(synth: Synthesizer) -> dict:
    return {
        "X-Sample-Rate": str(synth.sample_rate),
        "X-Model": synth.model_id,
        "Cache-Control": "no-store",
    }


# --------------------------------------------------------------------------- #
# Metadata endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health(
    settings: Settings = Depends(get_settings),
    manager: EngineManager = Depends(get_manager),
):
    return {
        "status": "ok",
        "backend": settings.backend,
        "available_backends": available_backends(),
        "default_model": settings.model_id,
        "models": manager.available,
        "loaded": list(manager._synths.keys()),
        "sample_rate": settings.sample_rate,
    }


@app.get("/v1/models")
async def models(manager: EngineManager = Depends(get_manager)):
    # OpenAI-shaped list so tooling that expects it keeps working.
    return {
        "object": "list",
        "data": [{"id": m, "object": "model"} for m in manager.available],
        "default": manager.default_model,
    }


@app.get("/v1/voices")
async def voices(settings: Settings = Depends(get_settings)):
    # Reflect the active backend's capabilities. Mock accepts anything, so it
    # returns empty lists.
    return engine_class(settings.backend).capabilities()


# --------------------------------------------------------------------------- #
# Native endpoints
# --------------------------------------------------------------------------- #
@app.post("/v1/tts", dependencies=[Depends(require_auth)])
async def tts(req: TTSRequest, manager: EngineManager = Depends(get_manager)):
    """Non-streaming synthesis; returns a complete audio file."""
    synth = await resolve_synth(manager, req.model)
    try:
        pcm = await synth.synthesize_bytes(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        # e.g. the model doesn't support the requested mode.
        raise HTTPException(status_code=400, detail=str(exc))
    body = pcm_to_wav(pcm, synth.sample_rate) if req.response_format == "wav" else pcm
    return Response(
        content=body,
        media_type=CONTENT_TYPES[req.response_format],
        headers=_stream_headers(synth),
    )


@app.post("/v1/tts/stream", dependencies=[Depends(require_auth)])
async def tts_stream(req: TTSRequest, manager: EngineManager = Depends(get_manager)):
    """Chunked streaming synthesis; audio starts arriving immediately."""
    synth = await resolve_synth(manager, req.model)
    return StreamingResponse(
        synth.stream_response(req, req.response_format),
        media_type=CONTENT_TYPES[req.response_format],
        headers=_stream_headers(synth),
    )


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoint: POST /v1/audio/speech
# --------------------------------------------------------------------------- #
@app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
async def openai_speech(
    body: OpenAISpeechRequest, manager: EngineManager = Depends(get_manager)
):
    """Drop-in for the OpenAI audio/speech API. Streams the response body."""
    req = TTSRequest(
        text=body.input,
        language=body.language,
        speaker=body.voice or None,   # empty -> backend default
        speed=body.speed,
        instruct=body.instructions,
        model=body.model,
        response_format=body.response_format,
    )
    synth = await resolve_synth(manager, req.model)
    return StreamingResponse(
        synth.stream_response(req, req.response_format),
        media_type=CONTENT_TYPES[req.response_format],
        headers=_stream_headers(synth),
    )


# --------------------------------------------------------------------------- #
# WebSocket endpoint: /v1/tts/ws  (bidirectional streaming)
# --------------------------------------------------------------------------- #
#
# Protocol (all client->server frames are JSON text; server->client audio is
# sent as binary frames, control as JSON text):
#
#   C->S {"type":"config", ...defaults...}          set session defaults
#   C->S {"type":"synthesize","text":"...","request_id":"1", ...overrides...}
#   C->S {"type":"cancel"}                           cancel the active synthesis
#   C->S {"type":"close"}                            end the session
#
#   S->C {"type":"ready","models":[...],"default_model":...}   on connect
#   S->C {"type":"start","request_id":...,"sample_rate":...,"model":...,"format":...}
#   S->C  <binary audio frame> ...                   PCM (or WAV incl. header)
#   S->C {"type":"end","request_id":...,"cancelled":bool}
#   S->C {"type":"error","request_id":...,"message":...}
#
# Sending and receiving run concurrently, so a client can stream text segments
# and receive audio at the same time, and cancel mid-utterance.
# --------------------------------------------------------------------------- #
_SESSION_FIELDS = (
    "model", "language", "speaker", "instruct",
    "ref_audio", "ref_text", "response_format",
)


def _build_req(session: dict, msg: dict) -> TTSRequest:
    data = {k: session[k] for k in _SESSION_FIELDS if k in session}
    for k in (*_SESSION_FIELDS, "text"):
        if k in msg and msg[k] is not None:
            data[k] = msg[k]
    return TTSRequest(**data)


async def _run_ws_synthesis(
    ws: WebSocket, synth: Synthesizer, req: TTSRequest,
    request_id, cancel: threading.Event,
) -> None:
    await ws.send_json({
        "type": "start", "request_id": request_id,
        "sample_rate": synth.sample_rate, "model": synth.model_id,
        "format": req.response_format,
    })
    try:
        if req.response_format == "wav":
            await ws.send_bytes(wav_header(synth.sample_rate))
        async for pcm in synth._pcm_chunks(req, cancel):
            await ws.send_bytes(pcm)
        await ws.send_json({
            "type": "end", "request_id": request_id,
            "cancelled": cancel.is_set(),
        })
    except Exception as exc:  # report, but keep the socket open
        await ws.send_json({
            "type": "error", "request_id": request_id, "message": str(exc),
        })


@app.websocket("/v1/tts/ws")
async def tts_ws(ws: WebSocket):
    settings = get_settings()
    manager: EngineManager = ws.app.state.manager

    # Auth: browsers can't set headers on a WS, so accept ?api_key= too.
    authz = ws.headers.get("authorization")
    key = ws.query_params.get("api_key")
    if not (_valid_token(authz, settings)
            or (settings.allowed_keys and key in settings.allowed_keys)
            or not settings.allowed_keys):
        await ws.close(code=1008)
        return

    await ws.accept()
    await ws.send_json({
        "type": "ready", "models": manager.available,
        "default_model": manager.default_model,
    })

    session: dict = {}
    active: asyncio.Task | None = None
    cancel: threading.Event | None = None

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type", "synthesize")

            if mtype == "config":
                for k in _SESSION_FIELDS:
                    if k in msg:
                        session[k] = msg[k]
                await ws.send_json({"type": "configured", "session": session})

            elif mtype in ("synthesize", "tts"):
                if active is not None and not active.done():
                    await ws.send_json({
                        "type": "error", "request_id": msg.get("request_id"),
                        "message": "busy: a synthesis is already in progress",
                    })
                    continue
                try:
                    req = _build_req(session, msg)
                    synth = await manager.get(req.model)
                except UnknownModelError as exc:
                    await ws.send_json({
                        "type": "error", "request_id": msg.get("request_id"),
                        "message": f"unknown model {exc.args[0]!r}",
                    })
                    continue
                except Exception as exc:  # validation / bad request fields
                    await ws.send_json({
                        "type": "error", "request_id": msg.get("request_id"),
                        "message": str(exc),
                    })
                    continue
                cancel = threading.Event()
                active = asyncio.create_task(
                    _run_ws_synthesis(ws, synth, req, msg.get("request_id"), cancel)
                )

            elif mtype == "cancel":
                if cancel is not None:
                    cancel.set()

            elif mtype == "close":
                break

            else:
                await ws.send_json({
                    "type": "error", "message": f"unknown message type {mtype!r}",
                })
    except WebSocketDisconnect:
        pass
    finally:
        if cancel is not None:
            cancel.set()
        if active is not None:
            try:
                await active
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)
