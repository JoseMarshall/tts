"""FastAPI application: native, OpenAI-compatible, and WebSocket TTS endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import (
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse

from .audio import CONTENT_TYPES, pcm_to_wav, wav_header
from .config import Settings, get_settings, get_sst_settings
from .engine import available_backends, engine_class, sst_available_backends, sst_engine_class
from .manager import EngineManager, SSTManager, UnknownModelError
from .schemas import OpenAISpeechRequest, OpenAISSTResponse, SSTRequest, TTSRequest
from .streaming import Synthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    sst_settings = get_sst_settings()

    # Validate TTS backend at startup.
    if settings.backend not in available_backends():
        raise RuntimeError(
            f"Unknown TTS_BACKEND {settings.backend!r}. "
            f"Available: {available_backends()}"
        )

    manager = EngineManager(settings)
    sst_manager = SSTManager(sst_settings)  # type: ignore[arg-type]

    app.state.tts_manager = manager
    app.state.sst_manager = sst_manager

    log.info("Starting TTS server (default=%s, backends=%s, catalog=%s)",
             manager.default_spec.key, manager.enabled_backends,
             [s.key for s in manager.catalog])
    log.info("SST defaults: backend=%s models=%s",
             sst_manager.default_spec.key, sst_manager.enabled_backends)

    await manager.preload_default()
    await sst_manager.preload_default()
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
def get_tts_manager(request: Request) -> EngineManager:
    return request.app.state.tts_manager


def get_sst_manager(request: Request) -> SSTManager:
    return request.app.state.sst_manager


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
    manager: EngineManager = Depends(get_tts_manager),
):
    return {
        "status": "ok",
        "default_backend": settings.backend,
        "default_model": manager.default_model,
        "enabled_backends": manager.enabled_backends,
        "installed_backends": available_backends(),
        "catalog": [{"id": s.model_id, "backend": s.backend} for s in manager.catalog],
        "loaded": list(manager._synths.keys()),
        # SST health ──────────────────────────────────────────────────────────
        "sst_default_model": app.state.sst_manager.default_spec.key,
        "sst_enabled_backends": app.state.sst_manager.enabled_backends,
        "sst_catalog": [{"id": s.model_id, "backend": s.backend}
                        for s in app.state.sst_manager.catalog],
    }


@app.get("/v1/models")
async def models(manager: EngineManager = Depends(get_tts_manager)):
    # OpenAI-shaped list, enriched with backend + selectable aliases so a client
    # can discover what to put in the request's "model" field.
    data = [{"id": s.model_id, "object": "model", "backend": s.backend}
            for s in manager.catalog]
    return {
        "object": "list",
        "data": data,
        "default": manager.default_model,
        "backends": manager.enabled_backends,  # each selectable as model="<name>"
    }


@app.get("/v1/voices")
async def voices(
    model: str | None = None,
    manager: EngineManager = Depends(get_tts_manager),
):
    # Capabilities depend on the model's backend. Defaults to the default model;
    # pass ?model=<id-or-backend> for a specific one.
    try:
        spec = manager.resolve(model)
    except UnknownModelError:
        raise HTTPException(status_code=400, detail=f"Unknown model {model!r}")
    caps = engine_class(spec.backend).capabilities()
    caps["model"] = spec.model_id
    return caps


# --------------------------------------------------------------------------- #
# Native endpoints
# --------------------------------------------------------------------------- #
@app.post("/v1/tts", dependencies=[Depends(require_auth)])
async def tts(req: TTSRequest, manager: EngineManager = Depends(get_tts_manager)):
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
async def tts_stream(req: TTSRequest, manager: EngineManager = Depends(get_tts_manager)):
    """Chunked streaming synthesis; audio starts arriving immediately."""
    synth = await resolve_synth(manager, req.model)
    # Admission control BEFORE the response starts — once StreamingResponse is
    # returned the status/headers are already sent and can't become a 503.
    if synth.at_capacity:
        raise HTTPException(status_code=503, detail="server busy: queue is full")
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
    body: OpenAISpeechRequest, manager: EngineManager = Depends(get_tts_manager)
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
    if synth.at_capacity:
        raise HTTPException(status_code=503, detail="server busy: queue is full")
    return StreamingResponse(
        synth.stream_response(req, req.response_format),
        media_type=CONTENT_TYPES[req.response_format],
        headers=_stream_headers(synth),
    )


# --------------------------------------------------------------------------- #
# SST introspection                                                             #
# --------------------------------------------------------------------------- #


@app.get("/v1/sst/models")
async def sst_models(
    manager: SSTManager = Depends(get_sst_manager),
):
    """Discover which SST backends and models are available."""
    data = [{"id": s.model_id, "object": "model", "backend": s.backend}
            for s in manager.catalog]
    return {
        "object": "list",
        "data": data,
        "default": manager.default_model,
        "backends": manager.enabled_backends,
    }


@app.get("/v1/sst/voices")
async def sst_voices(
    model: str | None = None,
    manager: SSTManager = Depends(get_sst_manager),
):
    """Capabilities of the active SST backend/model (languages, formats, …)."""
    try:
        spec = manager.resolve(model)
    except UnknownModelError:
        raise HTTPException(status_code=400, detail=f"Unknown model {model!r}")
    caps = sst_engine_class(spec.backend).capabilities()
    caps["model"] = spec.model_id
    return caps


# --------------------------------------------------------------------------- #
# SST native endpoints                                                          #
# --------------------------------------------------------------------------- #


@app.post("/v1/sst", dependencies=[Depends(require_auth)])
async def sst(
    body: dict = Body(...),
    manager: SSTManager = Depends(get_sst_manager),
):
    """Native speech-to-text endpoint.  Accepts ``audio`` (base64-encoded) + ``model``."""
    raw_audio_val = body.get("audio", "")
    if isinstance(raw_audio_val, str):
        import base64 as _b64
        padding = "=" * (-len(raw_audio_val) % 4)
        raw_audio = _b64.urlsafe_b64decode(raw_audio_val + padding)
    else:
        raw_audio = bytes(raw_audio_val) if raw_audio_val else b""

    if not raw_audio:
        raise HTTPException(status_code=400, detail="'audio' field is required (base64)")

    sst_model = body.get("model")
    lang = body.get("language")

    try:
        synth = await manager.get(sst_model)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {exc.args[0]!r}. Available: {manager.available}",
        )

    # Admission control.
    if synth.at_capacity:  # type: ignore[attr-defined]
        raise HTTPException(status_code=503, detail="server busy: queue is full")

    text = await synth.transcribe(raw_audio)  # type: ignore[attr-defined]
    fmt = body.get("response_format", "text")

    if fmt == "segments":
        return {"text": text, "segments": [s for s in text.split()]}
    return {"text": text}


@app.post("/v1/sst/stream", dependencies=[Depends(require_auth)])
async def sst_stream(
    body: dict = Body(...),
    manager: SSTManager = Depends(get_sst_manager),
):
    """Streaming transcription.  Yields text segments as JSON frames."""
    raw_audio_val = body.get("audio", "")
    if isinstance(raw_audio_val, str):
        import base64 as _b64
        padding = "=" * (-len(raw_audio_val) % 4)
        raw_audio = _b64.urlsafe_b64decode(raw_audio_val + padding)
    else:
        raw_audio = bytes(raw_audio_val) if raw_audio_val else b""

    if not raw_audio:
        raise HTTPException(status_code=400, detail="'audio' field is required (base64)")

    sst_model = body.get("model")

    try:
        synth = await manager.get(sst_model)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {exc.args[0]!r}. Available: {manager.available}",
        )

    if synth.at_capacity:  # type: ignore[attr-defined]
        raise HTTPException(status_code=503, detail="server busy: queue is full")

    async def segment_stream():
        yield json.dumps({"type": "start", "model": synth.model_id}).encode() + b"\n"  # type: ignore[attr-defined]
        try:
            async for seg in synth.stream_text(raw_audio):  # type: ignore[attr-defined]
                if seg.strip():
                    yield (json.dumps({"type": "segment", "text": seg}) + "\n").encode()
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}).encode() + b"\n"
        yield json.dumps({"type": "end"}).encode() + b"\n"

    return StreamingResponse(segment_stream(), media_type="application/x-ndjson")


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
async def openai_sst(
    request: Request,
    manager: SSTManager = Depends(get_sst_manager),
):
    """OpenAI-compatible transcription endpoint (drop-in).

    Accepts ``multipart/form-data`` with a ``file`` field (audio) and optional
    ``model``, ``language`` fields.  Mirrors the OpenAI ``/v1/audio/transcriptions`` API.
    """
    form = await request.form()
    audio_bytes = b""
    file_obj = form.get("file")
    if isinstance(file_obj, UploadFile):
        audio_bytes = await file_obj.read()
    elif isinstance(file_obj, bytes) and file_obj:
        audio_bytes = file_obj

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="'file' field is required (audio)")

    lang = form.get("language")
    fmt = form.get("response_format", "text")
    sst_model = None
    for k in ("model",):
        val = form.get(k)
        if val:
            sst_model = val

    try:
        synth = await manager.get(sst_model)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {exc.args[0]!r}. Available: {manager.available}",
        )

    if synth.at_capacity:  # type: ignore[attr-defined]
        raise HTTPException(status_code=503, detail="server busy: queue is full")

    text = await synth.transcribe(audio_bytes)  # type: ignore[attr-defined]

    if fmt == "json" or fmt == "verbose_json":
        return OpenAISSTResponse(text=text)
    return Response(content=text, media_type="text/plain")


# --------------------------------------------------------------------------- #
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
    manager: EngineManager = ws.app.state.tts_manager

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


# --------------------------------------------------------------------------- #
# WebSocket endpoint: /v1/sst_ws  (streaming audio-in → text-out)
# --------------------------------------------------------------------------- #
#
# Protocol (all text frames are JSON; binary frames carry raw PCM audio):
#
#   Client -> Server messages ------------------------------------------------ #
#   {"type":"init","model":"mock","language":"en"}     optional config        #
#   {"type":"start"}                                    beginning of segment  #
#   {"type":"chunk","data":"<base64_audio>"}           base64 audio chunk    #
#   ["binary frame"]                                  raw PCM bytes (eff.)  #
#   {"type":"flush"}                                  start transcription     #
#   {"type":"cancel"}                                 cancel in-progress       #
#   {"type":"close"}                                  end session              #
#                                                                               #
#   Server -> Client messages -------------------------------------------------- #
#   {"type":"ready","models":[...],"default_model":...} on connect             #
#   {"type":"segment","index":N,"text":"..."}         real-time segment        #
#   {"type":"done","count":N,"full_text":"..."}       transcription complete   #
#   {"type":"error","message":"..."}                  error or validation      #
# --------------------------------------------------------------------------- #

import base64 as _b64  # local alias — avoids per-call import in the handler


async def _run_ws_transcription(
    ws: WebSocket,
    transcriber,                # Transcriber instance from SSTManager
    audio_chunks: list[bytes],   # accumulated audio bytes
    language: str | None = None,
) -> tuple[int, str]:
    """Run streaming transcription and yield segment frames to ``ws``.

    Returns ``(count, full_text)`` after the engine finishes producing segments.
    """
    import asyncio as _ai  # local alias for type-hint simplicity
    count = 0
    parts: list[str] = []
    cancelled = False
    try:
        async for seg in transcriber.stream_text(
            b"".join(audio_chunks), language=language,
        ):  # type: ignore[attr-defined, union-attr]
            if cancelled:
                break
            count += 1
            parts.append(seg)
            await ws.send_json({
                "type": "segment", "index": count - 1, "text": seg,
            })
    except Exception as exc:
        pass
    return (count, " ".join(parts))


@app.websocket("/v1/sst_ws")
async def sst_ws_endpoint(ws: WebSocket):
    """Real-time Speech-to-Text via WebSocket.

    Clients stream audio chunks to the server and receive transcribed text
    segments concurrently (as soon as any engine produces them).
    """
    settings = get_settings()
    manager: SSTManager = ws.app.state.sst_manager

    # Auth (WS can't set Headers, so accept ?api_key=)
    authz = ws.headers.get("authorization")
    key = ws.query_params.get("api_key")
    if not (_valid_token(authz, settings)
            or (settings.allowed_keys and key in settings.allowed_keys)
            or not settings.allowed_keys):
        await ws.close(code=1008)
        return

    await ws.accept()

    # ---- per-session state ----------------------------------------------------
    await ws.send_json({
        "type": "ready",
        "models": manager.available,
        "default_model": manager.default_model,
        "sample_rate": 16000,
    })

    audio_buf: list[bytes] = []
    session: dict = {"model": None, "language": getattr(settings, "sst_default_language", "")}
    active_task: asyncio.Task | None = None
    segment_count: int = 0      # segments for the current transcription

    try:
        while True:
            kind = await ws.receive_type()

            if isinstance(kind, WebSocketDisconnect):
                break

            if isinstance(kind, bytes) or isinstance(kind, bytearray):
                # --- Binary frame → raw PCM chunk --------------------------------
                data = kind if isinstance(kind, (bytes, bytearray)) else bytes(kind)   # type: ignore[arg-type]
                audio_buf.append(data)
                continue

            if isinstance(kind, str):
                # Text frame is always JSON control message
                try:
                    msg = json.loads(kind)  # parse manually to avoid dependency on receive_json for all paths
                except (json.JSONDecodeError, TypeError):
                    await ws.send_json({"type": "error", "message": f"invalid JSON in text frame"})
                    continue

                mtype = msg.get("type")

                if mtype == "init":
                    session["model"] = msg.get("model")
                    session["language"] = msg.get("language")
                    await ws.send_json({"type": "configured"})

                elif mtype == "start":
                    await ws.send_json({"type": "start", "sample_rate": 16000})

                elif mtype == "chunk":
                    # Accumulate base64 audio chunk into buffer
                    data = msg.get("data", b"")
                    if isinstance(data, str) and data:
                        padding = "=" * (-len(data) % 4)
                        decoded = _b64.urlsafe_b64decode(data + padding)
                        audio_buf.append(decoded)
                    elif isinstance(data, (bytes, bytearray)):
                        audio_buf.append(bytes(data))

                elif mtype == "flush":
                    # Transcribe accumulated audio and send segments back
                    if not audio_buf:
                        await ws.send_json({"type": "error", "message": "no audio data to transcribe"})
                        continue
                    audio_bytes = b"".join(audio_buf)
                    audio_buf.clear()
                    segment_count += 1

                    # Get or create transcriber for the requested model
                    try:
                        model_name = msg.get("model") or session.get("model") or manager.default_model
                        transcriber = await manager.get(model_name)  # type: ignore[arg-type]
                    except Exception as exc:
                        await ws.send_json({"type": "error", "message": str(exc)})
                        continue

                    # Check capacity before starting
                    if transcriber.at_capacity:   # type: ignore[attr-defined]
                        await ws.send_json({"type": "error", "message": "server busy: queue is full"})
                        continue

                    # Run transcription as background task
                    active_task = asyncio.create_task(
                        _run_ws_transcription(
                            ws,
                            transcriber,       # Transcriber instance
                            [audio_bytes],     # list of audio chunks (already joined)
                            language=session.get("language"),
                        )
                    )

                elif mtype == "cancel":
                    # Cancel the active transcription task if it exists
                    if active_task is not None and not active_task.done():
                        active_task.cancel()
                        await ws.send_json({"type": "cancelled"})
                        active_task = None

                elif mtype == "close":
                    break

                else:
                    await ws.send_json({
                        "type": "error",
                        "message": f"unknown message type {mtype!r}",
                    })

    except WebSocketDisconnect:
        pass
    finally:
        if active_task is not None and not active_task.done():
            try:
                active_task.cancel()
                await active_task
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)