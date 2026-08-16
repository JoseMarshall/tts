# Proposal — VAD auto-flush on `/v1/stt_ws`

**Status:** implemented · **Affects:** new `app/vad.py`, `main.py` (`/v1/stt_ws` only),
`config.py`, `docs/websocket.md`
**Driven by:** hands-free turn-taking and barge-in — a voice agent that has to know
when the user *stopped* talking, and when they *started* talking over the reply.
**Follows:** [`timing-marks.md`](timing-marks.md), which named this the next most
valuable server change.

## The problem

`/v1/stt_ws` accumulates audio and transcribes only when the client says so:

```python
elif mtype == "flush":
    if not audio_buf:
        ...
    audio_bytes = b"".join(audio_buf)
```

That makes the *client* responsible for endpointing — deciding where the utterance
ends. A push-to-talk UI can do that, because a human released a button. Nothing else
can. A hands-free client has to either run its own VAD (and now there are two
opinions about where speech ended, one of them in JavaScript), flush on a fixed
timer (and cut words in half), or flush on every chunk (and pay a full transcription
per 100 ms of audio).

The server is the one holding the audio. It should be the one that notices the
silence.

## What's actually in the way

Three things, and only the first is about VAD at all.

**1. Transcription blocks the receive loop.** The handler awaits the whole
transcription inline, inside the same `while True` that reads frames:

```python
async for seg in transcriber_obj.stream_text(audio_bytes, ...):
    count += 1
    await ws.send_json({"type": "segment", ...})
```

While that runs, no `await ws.receive()` happens, so audio arriving mid-transcription
is buffered by the socket and processed late. Barge-in is not merely unimplemented —
it is unreachable from this shape. The transcription has to move to a task, and the
receive loop has to stay hot.

**2. The buffer is unbounded.** `audio_buf.append(...)` has no ceiling. A client that
connects and streams without ever sending `flush` grows a list of `bytes` until the
process dies. `STT_MAX_INPUT_SECONDS` exists in `config.py` and is read by nothing;
the per-engine `MAX_INPUT_SECONDS` only truncates at the point of transcription,
long after the memory was spent. Auto-flush fixes this incidentally — a hard
`max_utterance` cap means the buffer can no longer grow without bound — and that
is worth as much as the feature.

**3. `_run_ws_transcription()` is dead code.** Defined at `main.py:599`, called from
nowhere, and it swallows every exception into a bare `except Exception: pass`. The
work below either uses it or deletes it; leaving a second, subtly-different copy of
the transcription loop next to the live one is how the two drift.

## Proposed VAD API

A detector is small and boring, and there are three reasonable ones. Rather than
pick a winner in the handler, register them the way engines already register:

```python
class VAD(abc.ABC):
    NAME: str = ""
    SAMPLE_RATE: int = 16000
    FRAME_SAMPLES: int = 512      # the window size this detector demands

    @abc.abstractmethod
    def speech_prob(self, frame: np.ndarray) -> float:
        """P(speech) for exactly FRAME_SAMPLES of float32 mono audio."""

    def reset(self) -> None:      # clear recurrent state between utterances
        pass
```

`speech_prob` rather than `is_speech` because Silero returns a probability and
throwing it away to recover a threshold in the caller is a lossy round-trip. Energy
and WebRTC detectors return `0.0`/`1.0` and nothing is worse for it.

Three implementations, mirroring how backends are bundled:

- **`SileroVAD` (default).** `pip install silero-vad`, ~2 MB, runs on CPU in
  well under real time. Accurate on noisy and far-field input, which is the case
  that makes energy detectors embarrassing.
- **`EnergyVAD`.** RMS plus zero-crossing rate, no dependencies. This is the path
  tests and the `mock` backend take, so the suite keeps running on a machine with
  nothing installed — the same reason `MockEngine` exists.
- **`WebrtcVAD`.** `pip install webrtcvad`. Included because it is what a lot of
  telephony pipelines already standardised on, not because it is better.

### Turn detection is separate from speech detection

The detector answers "is this 32 ms speech?". Nothing usable comes from that answer
alone — raw VAD output flickers, and flushing on the first quiet frame chops the
stop consonant off every word ending in /t/. The state machine on top is where the
behaviour actually lives:

```python
class TurnDetector:
    """VAD + hysteresis: turns per-frame probabilities into turn boundaries."""

    def feed(self, pcm: bytes) -> list[TurnEvent]: ...
```

with four knobs that matter:

| Knob | Default | Why |
|---|---|---|
| `speech_ms` | 120 | consecutive speech before declaring onset — rejects clicks and door slams |
| `silence_ms` | 700 | trailing silence that ends the turn — the single knob users will actually tune |
| `pre_roll_ms` | 300 | audio retained *before* onset, so the first phoneme survives |
| `max_utterance_s` | 30 | hard cap: force a flush, and bound the buffer |

`pre_roll_ms` is the one that is invisible until it is missing. Speech is declared
120 ms after it began; without a pre-roll ring buffer, every utterance starts with
"…ello" instead of "Hello", and the transcript is wrong in a way that reads as a
model problem rather than a buffering one.

## Wire protocol

Additive frames on `/v1/stt_ws`, under the forward-compatibility rule
`docs/websocket.md` already states (clients ignore unknown `type` values):

```jsonc
// S->C, the moment speech onset is confirmed. THE barge-in signal.
{"type":"speech_start","t":1.28}
// S->C, trailing silence hit; transcription of this turn is starting.
{"type":"speech_end","t":3.94,"duration":2.66,"reason":"silence"}
```

Then the existing `segment` / `done` frames follow, byte-for-byte identical to what
an explicit `flush` produces. That is the point: auto-flush is a new *trigger* for
the existing pipeline, not a second pipeline. `reason` is `"silence"`, `"max_utterance"`,
or `"client_flush"`, so a client can tell a natural ending from a truncation.

`t` is seconds since the session's first audio sample — session-relative, not
turn-relative, because a client correlating VAD events against its own playback
clock needs a monotonic session timeline.

Per-session control on `init`, and discovery on `ready`:

```jsonc
// C->S
{"type":"init","model":"whisper","vad":{"enabled":true,"silence_ms":500}}
// S->C
{"type":"ready","models":[...],"vad":{"available":true,"backend":"silero","enabled":false,
                                      "silence_ms":700,"max_utterance_s":30}}
```

### Off by default, unlike marks

Timing marks defaulted to *on* because they added an ignorable frame type and
changed nothing else. This is the opposite: auto-flush changes when transcription
happens, which is session semantics. A client written against the current handler —
buffer, flush when the user releases the button — would start receiving `done`
frames it never asked for, mid-utterance, and the second `flush` would find an
empty buffer and get an error.

So: `STT_VAD_AUTO_FLUSH=0` by default, opt in with `{"vad":{"enabled":true}}` on
`init`, and let an operator flip the server-wide default to `1` when every client
on that deployment wants it. Explicit `flush` keeps working either way and ends the
current turn immediately.

## Configuration

```
STT_VAD=silero               # silero | energy | webrtc — which detector to build
STT_VAD_AUTO_FLUSH=0         # server-wide default for new sessions
STT_VAD_THRESHOLD=0.5        # speech_prob above this counts as speech
STT_VAD_SPEECH_MS=120
STT_VAD_SILENCE_MS=700
STT_VAD_PRE_ROLL_MS=300
STT_VAD_MAX_UTTERANCE_S=30
```

`STT_VAD=silero` names the default detector, but nothing imports `silero_vad` until
a session actually enables VAD — the same lazy-import discipline every engine
already follows, so the server still starts on a machine with no optional packages
installed. When the import fails the error names the pip package and says
`STT_VAD=energy` is the zero-dependency path.

## The one subtle part

**The VAD must not run on the event loop, and must not queue behind the GPU.**

Silero is a torch model. Calling it inline in the receive loop means every audio
frame from every session serialises through a synchronous forward pass on the same
thread that is supposed to be reading sockets — the exact stall this endpoint
already has for transcription, reintroduced at 30× the frequency.

It also must not go through `Transcriber`'s semaphore. That semaphore exists to
keep one ASR model off another's VRAM; putting a 2 MB detector behind it means
speech onset is not detected until the *previous* turn's transcription finishes,
which is precisely when barge-in needs it. Silero runs on CPU, in a thread, outside
the GPU lock. It is small enough that this is free.

And the frames must be re-aligned. Silero wants exactly 512 samples at 16 kHz;
clients send whatever their audio worklet produced. A re-framer accumulates the
byte stream into fixed windows and carries the remainder — get this wrong and the
detector sees garbage at every chunk boundary, which shows up as a threshold that
"needs tuning" rather than as a bug.

## Non-goals

- **Semantic endpointing.** Knowing that "my card number is four two —" is
  grammatically unfinished requires a language model in the audio path. That is a
  different, much larger proposal, and 700 ms of silence covers most of the value.
- **Diarization.** Who is speaking is a separate model and a separate proposal.
- **VAD on the HTTP endpoints.** `/v1/stt` and `/v1/stt/stream` receive a complete
  buffer; there is no turn to detect. Trimming leading and trailing silence there
  is a different optimisation.
- **Server-side barge-in orchestration.** The server emits `speech_start`. Deciding
  to stop the avatar's mouth and cancel the in-flight TTS is the client's call — it
  owns the playback buffer and knows what has actually reached the speaker. Same
  reasoning that kept visemes out of the marks proposal.

## Open questions (resolved during implementation)

1. **Does the pre-roll interact with `speech_end`?** A turn's audio is
   `pre_roll + speech + trailing silence`. The implementation keeps the trailing
   silence — sending it to the ASR is mildly wasteful and harmless, while
   trimming it risks clipping a final consonant. Revisit with measurements, not
   before.
2. **What happens to a `speech_start` that arrives while a transcription is still
   running?** Turns go onto a queue drained by one worker per session, so the
   receive loop stays hot and the event goes out immediately while the previous
   turn finishes and delivers its `done`. Ordering within a session is preserved.
   `cancel` now signals queued *and* in-flight turns, so a client that has moved
   on can drop them rather than waiting.
3. **Should `STT_MAX_INPUT_SECONDS` be wired up at the same time?** Left alone.
   `STT_VAD_MAX_UTTERANCE_S` bounds the buffer on the socket path, which was the
   actual hazard; `STT_MAX_INPUT_SECONDS` remains dead config on the HTTP path
   and should be either wired up or deleted in its own change.

## What shipped, and what didn't

Implemented as described, with two deliberate deviations:

- **`flush` outside a turn returns the pre-roll rather than an error.** The
  proposal left this open. An explicit flush is the client saying "transcribe
  what I sent", and an empty transcript answers that more honestly than
  "no audio data to transcribe" does when the client demonstrably sent audio.
  Still bounded, because the pre-roll ring is capped.
- **A detector that fails to load degrades the session, not the endpoint.** One
  `error` frame, VAD off, manual `flush` still works. An operator who sets
  `STT_VAD=silero` without installing it should lose auto-flush, not the socket.

Also folded in, because the rework touched them: `_run_ws_transcription()` (dead
since it was written, and it swallowed every exception) is gone, replaced by the
worker that is actually called; and `cancel` does something, having previously
tested a variable that was never assigned.
