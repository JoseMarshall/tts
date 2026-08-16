# Proposal — Timing marks alongside streamed audio

**Status:** proposed · **Affects:** `engine.py`, `streaming.py`, `main.py` (WS only),
`docs/websocket.md`
**Driven by:** a lip-synced 3D avatar client that needs to know *when* each sound
happens, not just what the audio is.

## The problem

`stream()` yields audio and nothing else:

```python
def stream(self, req: TTSRequest) -> Iterator[np.ndarray]:
    """Yield successive float32 mono audio chunks for ``req``."""
```

That is the right primitive for playing a clip, and the wrong one for animating a face.
A client that wants to move a mouth in time with the audio has three options, all bad:
run its own forced aligner over audio it just received, guess from the text, or drive the
jaw off an amplitude envelope. All three re-derive something the model already knew.

## What Kokoro already computes and we throw away

`KokoroEngine.stream()` reads one field off each result and discards the object:

```python
for result in pipeline(req.text, voice=voice, speed=speed):
    audio = getattr(result, "audio", None)
    ...
    yield frame
```

`KPipeline` yields a `Result` with rather more on it:

```python
@dataclass
class Result:
    graphemes: str
    phonemes: str
    tokens: Optional[List[en.MToken]] = None
    output: Optional[KModel.Output] = None      # .audio and .pred_dur
    text_index: Optional[int] = None
```

`KPipeline.join_timestamps(tokens, pred_dur)` walks the model's predicted per-frame
durations and writes `start_ts` / `end_ts` onto each `MToken`, which also carries that
token's `text` and `phonemes`. So the alignment is a **free byproduct of synthesis** —
already computed, already in memory, currently dropped one line before it could be used.

> Verify the field names and the `join_timestamps` arithmetic against the installed
> kokoro version before implementing. The fields above are from the current upstream
> source; nothing is installed in `.venv` today, so this has not been run.

Resolution note: `MToken` timestamps are per **token** (roughly per word) with that
token's phoneme string attached. True per-phoneme timing means subdividing with
`result.pred_dur` directly. Word-level plus phonemes is a large enough improvement to
ship first, and the finer version is additive.

## Proposed engine API

Keep `stream()` exactly as it is — required, audio-only, and the thing every existing
backend already implements. Add an optional marked variant with a default that costs
nothing:

```python
@dataclass
class Mark:
    kind: Literal["word", "phoneme"]
    text: str        # grapheme for word marks, the symbol itself for phoneme marks
    phonemes: str    # this token's phonemes; "" when kind == "phoneme"
    start: float     # seconds, relative to the start of THIS request
    end: float


class TTSEngine(abc.ABC):
    SUPPORTS_MARKS: bool = False

    @abc.abstractmethod
    def stream(self, req: TTSRequest) -> Iterator[np.ndarray]: ...

    def stream_marked(
        self, req: TTSRequest
    ) -> Iterator[tuple[np.ndarray, list[Mark]]]:
        """Audio plus any marks whose time range falls in that chunk."""
        for chunk in self.stream(req):
            yield chunk, []
```

Mock, Qwen and Dia inherit the default and are untouched. Kokoro overrides
`stream_marked()` and defines `stream()` in terms of it, so there is one code path and no
chance of the two drifting.

**The one subtle part.** `stream()` re-slices each Kokoro segment into fixed
`stream_chunk_samples` frames. Marks are timed against the *segment*, so the override has
to keep a running sample offset across segments and emit each mark with the frame whose
time range contains it. Get this wrong and timing drifts later into longer replies —
which is exactly where a lip-sync bug is most visible and hardest to attribute.

## Wire protocol

**WebSocket only.** A new server→client frame, sent immediately before the audio frame
whose range it covers:

```jsonc
{"type":"marks","request_id":"3","marks":[
  {"kind":"word","text":"particularly","phonemes":"pɑɹˈtɪkjəlɚli","start":0.41,"end":1.02}
]}
```

HTTP chunked streaming keeps its current shape. There is no clean side channel in a
chunked body, and every client that wants marks wants a socket anyway — it is already how
you get per-sentence `synthesize` and `cancel`.

Two supporting changes:

- `capabilities()` and `GET /v1/voices` advertise `supports_marks`, so a client can
  decide without probing.
- `docs/websocket.md` states as a forward-compatibility rule that clients **must ignore
  unknown `type` values**. That rule is what lets marks default to on.

Default on where supported: a few hundred bytes per sentence against a client that has to
opt in to something it cannot discover. `TTS_EMIT_MARKS=0` for operators who want the
bytes back.

## Non-goals

- **Visemes.** The server emits phonemes; mapping them to mouth shapes is the client's
  business, because VRM, ARKit and every other rig disagree about the vocabulary. Putting
  a viseme table in here would make a general-purpose speech server carry one particular
  avatar's opinions.
- **Forced alignment for engines that lack timings.** Qwen and Dia report
  `supports_marks: false` and that is the end of it. Bolting an aligner onto the response
  path would add a second model, GPU contention with the synthesis it is aligning, and
  latency to the thing this server exists to keep fast.
- **Per-phoneme resolution in v1.** `pred_dur` is right there when word-level proves
  insufficient.

## Open questions

1. **Does `speed` scale the timestamps?** Kokoro applies `speed` inside the pipeline. If
   `start_ts` reflects pre-speed timing, marks must be divided by `speed` before they go
   on the wire. Needs one experiment at `speed=1.5`.
2. **Segment boundaries and silence.** Kokoro emits nothing for some segments (the
   existing `if audio is None: continue`). Confirm the sample offset advances correctly
   when a segment contributes no audio, or everything after the first pause is late.
3. **Cancellation mid-sentence.** Marks already sent describe audio the client may never
   play. Client-side concern, but `docs/websocket.md` should say so explicitly rather
   than leaving each client to discover it.

## Not in this proposal

- **VAD auto-flush on `/v1/sst_ws`.** The handler buffers audio and transcribes only on
  an explicit `flush`, so a client cannot do hands-free turn-taking or barge-in. This is
  the next most valuable server change after marks and deserves its own proposal — it
  alters session semantics rather than adding a frame type.
- **Concurrent generation.** One model, serialised behind `Semaphore(1)`, `503` past
  `TTS_MAX_QUEUE`. A real ceiling, but replicas behind a load balancer solve it without
  redesigning anything, and nothing here makes it worse.
