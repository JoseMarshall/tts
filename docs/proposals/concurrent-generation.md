# Proposal — Concurrent generation via an in-process engine pool

**Status:** implemented · **Affects:** `streaming.py`, `manager.py`, `engine.py`,
`config.py`, `main.py` (`/health` only), `docs/architecture.md`,
`docs/deployment.md`
**Driven by:** a single 82M-parameter model holding a whole GPU at ~4% utilisation
while requests queue behind `Semaphore(1)`.
**Follows:** [`timing-marks.md`](timing-marks.md), which listed this as a real
ceiling and argued replicas behind a load balancer solve it. They do. They also
cost a whole process, a whole CUDA context, and a whole copy of every *other*
model the server hosts, to add one more Kokoro.

## The problem

`Synthesizer` serialises every generation for a model:

```python
self._gpu_lock = asyncio.Semaphore(1)   # one generation at a time
self._inflight = 0                        # queued + running requests
```

That is correct and necessary when one model instance saturates the device. It is
pure waste when it does not. Kokoro-82M is ~330 MB; a 24 GB card fits dozens, and
a single stream of it leaves the SMs mostly idle between short forward passes.
Under concurrent load the server does not get slower per request — it gets
*serialised*, so the tenth caller waits for nine full syntheses before hearing a
byte, and then `TTS_MAX_QUEUE` starts returning 503 on a GPU that was never busy.

The horizontal-scaling answer is right for capacity and wrong for granularity.
Replicas scale everything at once: a second process to double Kokoro throughput
also loads a second Dia (1.6B, and idle), or requires per-backend deployments and
a router that knows which is which. The unit of scaling should be the model, not
the process.

## Proposed change

Keep everything about how a generation runs. Change how many can run.

```python
class EnginePool:
    """N interchangeable instances of one model; a generation borrows one."""

    def __init__(self, build: Callable[[], TTSEngine], size: int):
        self._build, self._size = build, size
        self._free: asyncio.Queue[TTSEngine] = asyncio.Queue()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[TTSEngine]:
        engine = await self._free.get()
        try:
            yield engine
        finally:
            self._free.put_nowait(engine)
```

`Synthesizer` swaps its semaphore for the pool. The queue *is* the semaphore —
`acquire()` blocks when every replica is busy, exactly as `Semaphore(1)` blocks
today — so admission control, `_inflight`, cancellation and the worker-thread
bridge are untouched:

```python
async with self._pool.acquire() as engine:
    worker = threading.Thread(target=self._run_blocking, args=(engine, req, q, cancel), ...)
```

The one required change inside `_run_blocking` is that it takes the engine as an
argument instead of reading `self.engine`. That is the whole diff in spirit: the
Synthesizer stops owning *an* engine and starts owning *a set of* them.

`TTS_ENGINE_REPLICAS=1` is the default and reproduces today's behaviour exactly —
one instance, one at a time, same 503 threshold. Nothing changes for anyone who
does not opt in.

## Metadata has to come from somewhere

`Synthesizer.sample_rate` and `.model_id` read `self.engine`. With a pool they read
a designated first replica, which is fine because replicas are identical — except
where they aren't:

```python
# QwenEngine._raw_stream, the non-streaming fallback path
wavs, sr = method(**kwargs)
self.sample_rate = int(sr) or self.sample_rate
```

That mutates per-instance state from inside a generation. With one instance it is
invisible; with four, replica 3 can be at 24 kHz while `Synthesizer.sample_rate` —
and the `X-Sample-Rate` header, and the WAV header already sent — say something
else. This is a latent bug today (the header is written before the mutation
happens) that a pool turns into an intermittent one. Fix it in place: resolve the
sample rate once, at construction, and never write it mid-stream.

## The one subtle part

**Not every backend is safe to instantiate N times.**

Two Python objects are not two independent models if they share a global underneath.
The concrete hazard is Kokoro: `misaki` phonemizes through **espeak-ng**, a C
library with process-global state, via a binding that is not documented as
thread-safe. Four `KPipeline` instances are four Python objects sharing one espeak
context, and concurrent phonemization is where that would surface — intermittently,
as wrong phonemes rather than a crash, which is the worst possible failure mode
because it looks like a model quality problem.

So replication is opt-in *per backend*, declared the way `SUPPORTS_MARKS` already
is:

```python
class TTSEngine(abc.ABC):
    SUPPORTS_REPLICAS: bool = False    # can N instances run concurrently?
```

`MockEngine` sets it `True` immediately. The real backends each need one
verification pass before they flip, and until then `TTS_ENGINE_REPLICAS>1` on a
backend that has not opted in logs a warning and runs with one replica rather than
silently corrupting output. For Kokoro specifically the likely resolution is a
module-level lock around phonemization only — the G2P step is microseconds next to
the forward pass, so serialising just that costs almost nothing and leaves the
expensive part parallel.

> To verify before flipping any backend: whether `misaki`'s espeak-ng binding
> holds the GIL across the C call, and whether `kokoro` shares any module-level
> model or voice cache between `KPipeline` instances. Nothing is installed in
> `.venv` today, so this has not been run.

## Building N of them

Loading four models at first request means the first caller waits four times as
long, which trades a throughput problem for a latency problem. `EngineManager.get()`
builds replica 1, returns as soon as it is usable, and fills the rest in the
background:

```python
self._synths[spec.key] = Synthesizer(pool, self.settings)   # ready with 1
asyncio.create_task(pool.fill())                              # rest arrive later
```

The pool serves whatever is free, so a partially-filled pool is simply a smaller
pool. `warmup()` runs per replica, because a cold instance pays first-call
compilation whichever one a request lands on.

## Configuration

```
TTS_ENGINE_REPLICAS=1        # instances per model; >1 needs SUPPORTS_REPLICAS
TTS_MAX_QUEUE=32             # unchanged, still absolute
```

`TTS_MAX_QUEUE` deliberately does not scale with replicas. It bounds how many
callers are *waiting*, and a caller does not care whether the wait is caused by one
busy replica or four — the queue depth a client will tolerate is a property of the
client, not of the server's internals.

`/health` grows `replicas` per catalog entry, because "why is this still slow" and
"did my replica setting take effect" are the same question, and VRAM multiplied by
N is not something an operator should have to infer from `nvidia-smi`.

## Non-goals

- **Batching.** Coalescing several requests into one forward pass is a bigger win
  than replication and a much larger change: it needs per-backend batch support,
  padding and a scheduler that trades latency for throughput. Replication is the
  version that requires no cooperation from the model.
- **Multi-GPU placement.** `TTS_ENGINE_DEVICES=cuda:0,cuda:1` to pin replica *i*
  to device *i* is an obvious follow-on and deliberately not in v1; get one device
  right first.
- **Replacing horizontal scaling.** Replicas inside a process do not give you
  failover, rolling restarts, or more than one machine. `docs/deployment.md` keeps
  its scale-out guidance and gains a paragraph on when to reach for which — this is
  for filling a GPU you already paid for, not for surviving one dying.
- **Concurrency on the STT side.** `Transcriber` has the identical
  `Semaphore(1)` and would take the identical treatment, but ASR models are large
  enough that a second instance rarely fits. Separate decision, separate evidence.

## Open questions

1. **Does replication actually help for the model people run?** *Still open, and
   it is the one that matters.* The machinery is in and proven concurrent by
   test, but no throughput measurement has been taken on a real backend — none
   of them declares `SUPPORTS_REPLICAS` yet, so today this ships as capability
   plus a default of 1. Before flipping Kokoro: requests/sec at
   `TTS_ENGINE_REPLICAS` 1, 2 and 4 on one card. If the curve is flat, the
   ceiling is somewhere other than serialisation and the right change is to find
   it instead.
2. **Where should the queue wait be observable?** Still open. A caller that waits
   4 s for a replica and 1 s for generation cannot tell those apart, and neither
   can the operator. Time-in-queue probably belongs in the response headers.
3. **Should `at_capacity` account for pool size?** Resolved: documented, not
   subtracted. `TTS_MAX_QUEUE` bounds requests *in flight* — queued plus running
   — and stays absolute, because the depth a caller tolerates is a property of
   the caller rather than of how many replicas happen to be loaded.

## What shipped, and what didn't

Implemented as described. `TTS_ENGINE_REPLICAS` defaults to `1`, which is
byte-for-byte the old `Semaphore(1)` path — the free-queue of a one-element pool
*is* that semaphore. Concurrency is verified with a barrier that two generations
must meet inside `stream()`, so a pool of one fails it and a pool of two passes;
that tests overlap rather than a counter.

**No real backend declares `SUPPORTS_REPLICAS` yet.** Only `mock` does. The
espeak-ng verification described above has not been run — nothing is installed
in `.venv` — so flipping Kokoro would be asserting something unverified about a
C library's thread safety, and the failure mode is wrong phonemes rather than a
crash. The flag, the warning path and the `/health` reporting are all in place
for whoever runs that check.

Two fixes folded in, because the pool turns both from latent into intermittent:

- **`QwenEngine` no longer mutates `self.sample_rate` mid-stream.** The WAV
  header and `X-Sample-Rate` are both written before that line ran, so the
  mutation could only make the engine disagree with what the client was already
  told — and, with replicas, with its own siblings. It now logs the mismatch and
  names the setting to fix it.
- **`Settings.max_queue` was missing entirely**, while `streaming.py` read it on
  every request. Restored.
