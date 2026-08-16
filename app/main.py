"""FastAPI application: native, OpenAI-compatible, and WebSocket TTS endpoints."""
from __future__ import annotations

import asyncio
import base64 as _b64
import json
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
# The base class starlette's form parser actually produces; fastapi.UploadFile
# subclasses it, so this catches both.
from starlette.datastructures import UploadFile as StarletteUploadFile

from .audio import CONTENT_TYPES, pcm_to_wav, wav_header
from .config import Settings, get_settings, get_stt_settings
from .engine import Mark, available_backends, engine_class, stt_available_backends, stt_engine_class
from .manager import EngineManager, STTManager, UnknownModelError
from .schemas import OpenAISpeechRequest, OpenAISTTResponse, STTRequest, TTSRequest
from .streaming import Synthesizer
from .vad import TurnDetector, available_vads, build_vad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    stt_settings = get_stt_settings()

    # Validate TTS backend at startup.
    if settings.backend not in available_backends():
        raise RuntimeError(
            f"Unknown TTS_BACKEND {settings.backend!r}. "
            f"Available: {available_backends()}"
        )

    manager = EngineManager(settings)
    stt_manager = STTManager(stt_settings)  # type: ignore[arg-type]

    app.state.tts_manager = manager
    app.state.stt_manager = stt_manager

    log.info("Starting TTS server (default=%s, backends=%s, catalog=%s)",
             manager.default_spec.key, manager.enabled_backends,
             [s.key for s in manager.catalog])
    log.info("STT defaults: backend=%s models=%s",
             stt_manager.default_spec.key, stt_manager.enabled_backends)

    await manager.preload_default()
    await stt_manager.preload_default()
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


def get_stt_manager(request: Request) -> STTManager:
    return request.app.state.stt_manager


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
        # Replicas actually built per loaded model. "Did my TTS_ENGINE_REPLICAS
        # setting take effect" and "why is this still serialised" are the same
        # question, and VRAM x N should not have to be inferred from nvidia-smi.
        "replicas": {k: s.pool.size for k, s in manager._synths.items()},
        # STT health ──────────────────────────────────────────────────────────
        "stt_default_model": app.state.stt_manager.default_spec.key,
        "stt_enabled_backends": app.state.stt_manager.enabled_backends,
        "stt_catalog": [{"id": s.model_id, "backend": s.backend}
                        for s in app.state.stt_manager.catalog],
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
# STT introspection                                                             #
# --------------------------------------------------------------------------- #


@app.get("/v1/stt/models")
async def stt_models(
    manager: STTManager = Depends(get_stt_manager),
):
    """Discover which STT backends and models are available."""
    data = [{"id": s.model_id, "object": "model", "backend": s.backend}
            for s in manager.catalog]
    return {
        "object": "list",
        "data": data,
        "default": manager.default_model,
        "backends": manager.enabled_backends,
    }


@app.get("/v1/stt/voices")
async def stt_voices(
    model: str | None = None,
    manager: STTManager = Depends(get_stt_manager),
):
    """Capabilities of the active STT backend/model (languages, formats, …)."""
    try:
        spec = manager.resolve(model)
    except UnknownModelError:
        raise HTTPException(status_code=400, detail=f"Unknown model {model!r}")
    caps = stt_engine_class(spec.backend).capabilities()
    caps["model"] = spec.model_id
    return caps


# --------------------------------------------------------------------------- #
# STT native endpoints                                                          #
# --------------------------------------------------------------------------- #


def _form_str(form, key: str) -> str | None:
    """A multipart field as a string, or None if absent/empty/a file upload."""
    value = form.get(key)
    return value if isinstance(value, str) and value else None


def _decode_b64_audio(value: str) -> bytes:
    """Decode base64 audio, tolerating missing padding. Raises ``ValueError``.

    Accepts both the standard (``+/``) and URL-safe (``-_``) alphabets, because
    clients disagree about which they send. ``validate=True`` matters twice
    over: the permissive default silently DISCARDS characters outside the
    alphabet, so genuine garbage decoded to a few stray bytes instead of
    failing, and a standard-base64 payload containing ``+`` or ``/`` decoded to
    *corrupt audio* rather than being rejected or handled.
    """
    normalised = value.replace("-", "+").replace("_", "/")
    padding = "=" * (-len(normalised) % 4)
    return _b64.b64decode(normalised + padding, validate=True)


def _decode_audio_field(value: str) -> bytes:
    """HTTP wrapper: an undecodable ``audio`` body is a 400."""
    try:
        return _decode_b64_audio(value)
    except ValueError:      # binascii.Error is a ValueError subclass
        raise HTTPException(
            status_code=400, detail="'audio' is not valid base64",
        )


async def _resolve_transcriber(manager: STTManager, model: str | None):
    """Shared lookup + admission control for the native STT endpoints."""
    try:
        transcriber = await manager.get(model)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {exc.args[0]!r}. Available: {manager.available}",
        )
    if transcriber.at_capacity:  # type: ignore[attr-defined]
        raise HTTPException(status_code=503, detail="server busy: queue is full")
    return transcriber


@app.post("/v1/stt", dependencies=[Depends(require_auth)])
async def stt(
    req: STTRequest,
    manager: STTManager = Depends(get_stt_manager),
):
    """Native speech-to-text endpoint. Accepts ``audio`` (base64) + ``model``.

    Validation comes from :class:`STTRequest`, so a body missing ``audio``
    is a 422 (the schema was violated) rather than a hand-rolled 400. A
    *present but empty* or undecodable ``audio`` is still a 400 — the shape is
    right, the content is not.
    """
    raw_audio = _decode_audio_field(req.audio)
    if not raw_audio:
        raise HTTPException(status_code=400, detail="'audio' field is required (base64)")

    synth = await _resolve_transcriber(manager, req.model)
    synth.set_language(req.language)  # type: ignore[attr-defined]
    text = await synth.transcribe(raw_audio)  # type: ignore[attr-defined]

    if req.response_format == "segments":
        return {"text": text, "segments": text.split()}
    return {"text": text}


@app.post("/v1/stt/stream", dependencies=[Depends(require_auth)])
async def stt_stream(
    req: STTRequest,
    manager: STTManager = Depends(get_stt_manager),
):
    """Streaming transcription.  Yields text segments as JSON frames."""
    raw_audio = _decode_audio_field(req.audio)
    if not raw_audio:
        raise HTTPException(status_code=400, detail="'audio' field is required (base64)")

    synth = await _resolve_transcriber(manager, req.model)
    synth.set_language(req.language)  # type: ignore[attr-defined]

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
async def openai_stt(
    request: Request,
    manager: STTManager = Depends(get_stt_manager),
):
    """OpenAI-compatible transcription endpoint (drop-in).

    Accepts ``multipart/form-data`` with a ``file`` field (audio) and optional
    ``model``, ``language`` fields.  Mirrors the OpenAI ``/v1/audio/transcriptions`` API.
    """
    form = await request.form()
    audio_bytes = b""
    file_obj = form.get("file")
    # Starlette's form parser yields ITS OWN UploadFile, and fastapi.UploadFile
    # is a subclass of that — so testing against fastapi's silently missed every
    # real upload and returned "'file' field is required". Check the base class.
    if isinstance(file_obj, StarletteUploadFile):
        audio_bytes = await file_obj.read()
    elif isinstance(file_obj, (bytes, bytearray)) and file_obj:
        audio_bytes = bytes(file_obj)

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="'file' field is required (audio)")

    # form.get() returns str OR UploadFile; only strings mean anything here.
    lang = _form_str(form, "language")
    fmt = _form_str(form, "response_format") or "text"
    stt_model = _form_str(form, "model")

    synth = await _resolve_transcriber(manager, stt_model)
    synth.set_language(lang)  # type: ignore[attr-defined]
    text = await synth.transcribe(audio_bytes)  # type: ignore[attr-defined]

    if fmt in ("json", "verbose_json"):
        return OpenAISTTResponse(text=text)
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
#   S->C {"type":"start","request_id":...,"sample_rate":...,"model":...,"format":...,
#         "supports_marks":bool}
#   S->C {"type":"marks","request_id":...,"marks":[...]}       word timings (if any)
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


def _mark_payload(mark: Mark) -> dict:
    """Wire shape of one timing mark (times rounded to milliseconds)."""
    return {
        "kind": mark.kind,
        "text": mark.text,
        "phonemes": mark.phonemes,
        "start": round(mark.start, 3),
        "end": round(mark.end, 3),
    }


async def _run_ws_synthesis(
    ws: WebSocket, synth: Synthesizer, req: TTSRequest,
    request_id, cancel: threading.Event,
) -> None:
    supports_marks = synth.engine.SUPPORTS_MARKS
    await ws.send_json({
        "type": "start", "request_id": request_id,
        "sample_rate": synth.sample_rate, "model": synth.model_id,
        "format": req.response_format, "supports_marks": supports_marks,
    })
    # Marks frames go out only where the engine provides them and the
    # operator hasn't disabled them (TTS_EMIT_MARKS=0).
    emit = supports_marks and synth.settings.emit_marks
    try:
        if req.response_format == "wav":
            await ws.send_bytes(wav_header(synth.sample_rate))
        async for pcm, marks in synth.stream_marked_pcm(req, cancel):
            if emit and marks:
                await ws.send_json({
                    "type": "marks", "request_id": request_id,
                    "marks": [_mark_payload(m) for m in marks],
                })
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
# WebSocket endpoint: /v1/stt_ws  (streaming audio-in → text-out)
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
#   {"type":"ready","models":[...],"default_model":...,"vad":{...}} on connect  #
#   {"type":"speech_start","t":1.28}                  VAD onset (barge-in cue)  #
#   {"type":"speech_end","t":3.94,"reason":"silence"} turn ended; ASR starting  #
#   {"type":"segment","index":N,"text":"..."}         real-time segment        #
#   {"type":"done","count":N,"full_text":"..."}       transcription complete   #
#   {"type":"error","message":"..."}                  error or validation      #
# --------------------------------------------------------------------------- #

# Session-tunable VAD fields, and where each one's default comes from.
_VAD_FIELDS = {
    "enabled": "vad_auto_flush",
    "backend": "vad",
    "threshold": "vad_threshold",
    "speech_ms": "vad_speech_ms",
    "silence_ms": "vad_silence_ms",
    "pre_roll_ms": "vad_pre_roll_ms",
    "max_utterance_s": "vad_max_utterance_s",
}


def _vad_defaults(stt_settings) -> dict:
    """The server-wide VAD config a new session starts from."""
    cfg = {k: getattr(stt_settings, attr) for k, attr in _VAD_FIELDS.items()}
    cfg["enabled"] = bool(cfg["enabled"])
    return cfg


def _build_turn_detector(cfg: dict, sample_rate: int) -> TurnDetector:
    """Construct the detector + hysteresis. Heavy imports happen in here, so
    this belongs on a worker thread, not the event loop."""
    vad = build_vad(cfg["backend"], sample_rate)
    return TurnDetector(
        vad,
        sample_rate=sample_rate,
        threshold=float(cfg["threshold"]),
        speech_ms=int(cfg["speech_ms"]),
        silence_ms=int(cfg["silence_ms"]),
        pre_roll_ms=int(cfg["pre_roll_ms"]),
        max_utterance_s=float(cfg["max_utterance_s"]),
    )


async def _transcribe_turn(
    ws: WebSocket, manager: STTManager, session: dict, turn: dict,
) -> None:
    """Transcribe one turn's audio and send its frames.

    Identical output to what an explicit ``flush`` always produced — auto-flush
    is a new *trigger* for this pipeline, not a second pipeline.
    """
    model_name = turn.get("model") or session.get("model") or manager.default_model
    try:
        resolved = manager.resolve(model_name) if model_name else manager.default_spec
    except Exception:
        await ws.send_json({"type": "error", "message": f"unknown model {model_name!r}"})
        return

    try:
        transcriber = await manager.get(getattr(resolved, "model_id", model_name))
    except Exception as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        return

    if getattr(transcriber, "at_capacity", False):
        await ws.send_json({"type": "error", "message": "server busy: queue is full"})
        return

    lang = session.get("language")
    transcriber.set_language(lang)  # type: ignore[attr-defined]

    count = 0
    parts: list[str] = []
    cancel: threading.Event = turn["cancel"]
    try:
        async for seg in transcriber.stream_text(  # type: ignore[union-attr]
            turn["audio"], cancel, language=lang,
        ):
            count += 1
            parts.append(seg)
            await ws.send_json({"type": "segment", "index": count - 1, "text": seg})
    except Exception as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        return

    await ws.send_json({
        "type": "done", "count": count, "full_text": " ".join(parts),
        "reason": turn.get("reason", "client_flush"),
        "cancelled": cancel.is_set(),
    })


async def _stt_turn_worker(
    ws: WebSocket, manager: STTManager, session: dict, turns: asyncio.Queue,
) -> None:
    """Drain queued turns one at a time, off the receive loop.

    Transcription used to be awaited inline in the receive loop, so no frame
    was read while it ran — which is why barge-in was unreachable rather than
    merely unimplemented. One worker (not a task per turn) keeps a session's
    ``segment``/``done`` frames in turn order.
    """
    while True:
        turn = await turns.get()
        if turn is None:
            return
        try:
            await _transcribe_turn(ws, manager, session, turn)
        except (WebSocketDisconnect, RuntimeError):
            return  # socket gone; stop quietly
        except Exception:
            log.exception("STT turn failed")


@app.websocket("/v1/stt_ws")
async def stt_ws_endpoint(ws: WebSocket):
    """Real-time Speech-to-Text via WebSocket.

    Clients stream audio chunks to the server and receive transcribed text
    segments concurrently (as soon as any engine produces them).
    """
    settings = get_settings()
    manager: STTManager = ws.app.state.stt_manager

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
    stt_settings = get_stt_settings()
    sample_rate = stt_settings.sample_rate
    vad_cfg = _vad_defaults(stt_settings)

    await ws.send_json({
        "type": "ready",
        "models": manager.available,
        "default_model": manager.default_model,
        "sample_rate": sample_rate,
        "vad": {"available": vad_cfg["backend"] in available_vads(), **vad_cfg},
    })

    audio_buf: list[bytes] = []          # used only while VAD is off
    session: dict = {"model": None, "language": stt_settings.default_language or None}
    detector: TurnDetector | None = None
    turns: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(_stt_turn_worker(ws, manager, session, turns))
    cancels: list[threading.Event] = []   # cancel handles for queued/running turns

    async def _ensure_detector() -> TurnDetector | None:
        """Build the detector on first audio, on a thread (Silero loads a model).

        A failure here disables VAD for the session and falls back to manual
        `flush` rather than killing it — an operator who set STT_VAD=silero
        without installing it should lose auto-flush, not the endpoint.
        """
        nonlocal detector
        if detector is not None or not vad_cfg.get("enabled"):
            return detector
        loop = asyncio.get_running_loop()
        try:
            detector = await loop.run_in_executor(
                None, _build_turn_detector, dict(vad_cfg), sample_rate
            )
        except Exception as exc:
            vad_cfg["enabled"] = False
            log.warning("VAD unavailable, falling back to manual flush: %s", exc)
            await ws.send_json({"type": "error", "message": f"VAD unavailable: {exc}"})
        return detector

    def _enqueue(audio: bytes, reason: str, model: str | None = None) -> None:
        cancel = threading.Event()
        cancels.append(cancel)
        turns.put_nowait(
            {"audio": audio, "reason": reason, "model": model, "cancel": cancel}
        )

    async def _on_audio(pcm: bytes) -> None:
        det = await _ensure_detector()
        if det is None:                    # VAD off: buffer until `flush`
            audio_buf.append(pcm)
            return
        # The detector runs numpy (and, for Silero, torch) — keep it off the
        # event loop so other sessions' sockets stay responsive.
        loop = asyncio.get_running_loop()
        for ev in await loop.run_in_executor(None, det.feed, pcm):
            if ev.kind == "speech_start":
                await ws.send_json({"type": "speech_start", "t": round(ev.t, 3)})
            else:
                await ws.send_json({
                    "type": "speech_end", "t": round(ev.t, 3),
                    "duration": round(ev.duration, 3), "reason": ev.reason,
                })
                _enqueue(ev.audio, ev.reason)

    try:
        while True:
            # TestClient passes frames as either {"text": "..." } or {"bytes": b"..."}.
            frame = await ws.receive()  # type: ignore[no-untyped-call]

            if isinstance(frame, dict) and frame.get("type") == "websocket.disconnect":
                break

            if "bytes" in frame and frame["bytes"] is not None:
                await _on_audio(bytes(frame["bytes"]))  # type: ignore[arg-type]
                continue

            text = frame.get("text")
            if not text:
                continue

            try:
                msg = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                await ws.send_json({"type": "error", "message": "invalid JSON in text frame"})
                continue

            mtype = msg.get("type")

            if mtype == "init":
                session["model"] = msg.get("model")
                session["language"] = msg.get("language")
                vad_msg = msg.get("vad")
                if isinstance(vad_msg, dict):
                    for k in _VAD_FIELDS:
                        if k in vad_msg:
                            vad_cfg[k] = vad_msg[k]
                    vad_cfg["enabled"] = bool(vad_cfg["enabled"])
                    detector = None      # rebuilt on next audio with the new config
                await ws.send_json({"type": "configured", "vad": dict(vad_cfg)})

            elif mtype == "start":
                await ws.send_json({"type": "start", "sample_rate": sample_rate})

            elif mtype == "chunk":
                data = msg.get("data", b"")
                if isinstance(data, str):
                    try:
                        pcm = _decode_b64_audio(data)
                    except ValueError:
                        await ws.send_json({
                            "type": "error",
                            "message": "'data' is not valid base64",
                        })
                        continue
                    await _on_audio(pcm)
                elif isinstance(data, (bytes, bytearray)):
                    await _on_audio(bytes(data))

            elif mtype == "flush":
                # An explicit flush ends the current turn immediately, whether
                # or not VAD is running.
                event = detector.flush() if detector is not None else None
                if event is not None:
                    await ws.send_json({
                        "type": "speech_end", "t": round(event.t, 3),
                        "duration": round(event.duration, 3), "reason": event.reason,
                    })
                    _enqueue(event.audio, event.reason, msg.get("model"))
                elif audio_buf:
                    _enqueue(b"".join(audio_buf), "client_flush", msg.get("model"))
                    audio_buf.clear()
                else:
                    await ws.send_json(
                        {"type": "error", "message": "no audio data to transcribe"}
                    )

            elif mtype == "cancel":
                for cancel in cancels:
                    cancel.set()
                cancels.clear()
                while not turns.empty():           # drop turns not yet started
                    turns.get_nowait()
                audio_buf.clear()
                if detector is not None:
                    detector.flush()               # discard partial turn audio
                await ws.send_json({"type": "cancelled"})

            elif mtype == "close":
                break

            else:
                await ws.send_json({"type": "error", "message": f"unknown message type {mtype!r}"})
    except WebSocketDisconnect:
        pass
    finally:
        for cancel in cancels:
            cancel.set()
        turns.put_nowait(None)       # tell the worker to stop after its queue
        try:
            await asyncio.wait_for(worker, timeout=5)
        except (asyncio.CancelledError, Exception):
            worker.cancel()          # wedged or already gone; don't leak it


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)