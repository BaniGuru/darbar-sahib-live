# Increasing lag on the deployed two-core server

The reported server is running `b2f3f2d`. Its health JSON reports
`encoder_update_seconds=0.8`, adaptive acoustic word caching, and no
`recognition_mode`, block-size or BLAS fields. This is the older rolling
implementation. The 30-second playback delay and single-pass 4-second
recognizer were added later in `bead5f2` on `no-audio-fix`.

Increasing playback delay does not increase recognition throughput. In
rolling mode, growing processing lag can fill receive queues and delay
WebSocket pongs. The reported ping timeout then disconnects ASR; reconnecting
replays only recent context, which cannot recover the entire skipped span.
The later single-pass mode removes that repeated acoustic work. KenLM stays
enabled with 24 beams and the same scoring weights.

## Update the deployment

From the production checkout, preserving local changes:

```bash
cd /var/www/gurbani-asr
git fetch origin
git switch no-audio-fix
git pull --ff-only origin no-audio-fix
.venv/bin/python -m pip install -r requirements.txt
```

Restart the Uvicorn/systemd service after the update. Installing requirements
adds `threadpoolctl` if missing. Use these settings in the service's
Environment/EnvironmentFile, rather than only exporting them in an unrelated
shell:

```text
ASR_RECOGNITION_MODE=blocks
ASR_BLOCK_SECONDS=4
PLAYBACK_DELAY_SECONDS=10
ENABLE_LANGUAGE_MODEL=True
ASR_ENCODER_THREADS=2
ASR_BLAS_THREADS=1
```

These are the current two-core defaults. Keep the existing source URL and
service port. If `ASR_INTERNAL_WS_URL` points at a separate ASR process,
update/restart that process too; the startup runtime log identifies the
recognizer actually used by live forwarding.

Check both the checkout and running process:

```bash
git rev-parse --short HEAD
curl -fsS http://127.0.0.1:8000/asr/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/health | python3 -m json.tool
```

Adjust the port if Uvicorn uses another one. Expected ASR fields are
`recognition_mode=blocks`, `block_seconds=4`, `encoder_update_seconds=4`,
`normalization_lookahead_seconds=4`, `encoder_threads=2`, `blas_threads=1`,
`enable_language_model=true`, and `kenlm_order=3`. Health now includes a
source fingerprint (`build=stream-…`); public health also reports ASR mode
and block size. Startup logs identify the connected backend's build/mode.
A delayed frontend connected to rolling/legacy ASR emits a warning.

A roughly 4–5-second source-to-recognition lag is expected in block mode
because normalization deliberately waits for four seconds of future PCM.
It is separate from the listener's playback delay, now 10 seconds by default. New timing logs
include mode, processing cost and inference-lock wait, helping distinguish
work that exceeds the block's budget from intentional lookahead or another
inference request holding the lock.

## Two-core / four-GB validation

Tests used Docker CPU quota `200000 100000`, affinity `0,1`, a 4 GiB memory
limit, and no swap on the available x86-64 host. This is a real resource cap,
not a claim about the exact processor in the production server. KenLM was
enabled in every test; no beams, model weights or language weights were
reduced.

The original 52.0126875-second example and first 1,200 seconds of the human
verified recording produced exactly the same final tokens and word times
as the earlier four-core block run. All 65 example snapshots and all 1,500
recording snapshots also agree on captions, cache decisions and clocks.

| Metric, 20-minute recording | Result |
| --- | ---: |
| WER | 10.14% |
| CER excluding spaces | 7.47% |
| Missing reference words | 10 |
| Recognition runtime | 113.27 s |
| Real-time factor | 0.094 |
| Encoder calls / acoustic re-encodes | 300 / 0 |
| Retained PCM samples / encoder frames | 19,200,000 / 30,000 |
| Native container peak memory | 1.54 GiB |

WER/CER ignore only terminal sihari/onkar, punctuation and standalone verse
numbers, preserving internal vowels and boundaries. All recognized words
are scored against 572 reference words, including words in unannotated gaps.

A separate serial 120-second paced rapid passage used real HTTP/WebSocket
ASR with 100 ms packets. The production pankti matching handler and all
130,840 indexed rows ran in the same capped machine, in the receiving client
process. Matching produced 28 matched updates across five panktis. HTTP and
WebSocket final text, tokens, word times, cache events and sample/frame counts
agreed. This paced test includes recognition and matching; upstream FFmpeg
and browser playback are outside its scope. Full source/playback checks are
in [DELAYED_PLAYBACK.md](DELAYED_PLAYBACK.md).

| Delivery metric | Result |
| --- | ---: |
| Median / p95 / maximum lag | 4.426 / 4.779 / 4.974 s |
| First-half / last-half median lag | 4.431 / 4.430 s |
| Maximum block processing cost | 0.971 s |
| Maximum inference-lock wait | 0.000092 s |
| Combined container peak memory | 1.77 GiB |
| OOM events | 0 |

The lag did not grow during this paced passage. The 20-minute accuracy run
is an unpaced replay, not a 20-minute wall-time latency measurement. Exact
production performance still needs checking after the older deployment is
updated; the report does not treat the user's old-process logs as a test of
block mode.

## Disconnect fix

A transport can close during awaited inference before Starlette receives its
disconnect event. Uvicorn SansIO then rejects `websocket.send` with the
reported RuntimeError. ASR now exits cleanly on known closed-transport send
errors for ready, partial, error and final messages; close-after-final is
also safe. Other RuntimeErrors still propagate. This prevents the extra ASGI
traceback; it does not make an overloaded rolling recognizer faster. Ping
intervals/timeouts were not extended to hide an increasing backlog.

Six lifecycle tests cover closed transport, peer disconnect, final/close races,
and propagation of unrelated errors. A handler regression checks that ASR
stops sending/receiving after the reported error. An actual Uvicorn SansIO
test on the two-core container deliberately pauses 0.5 seconds before a
post-inference send, then closes the peer. The closed-send path was exercised
without an ASGI exception; a subsequent connection transcribed and finalized
6,400 samples. That artificial delay is only a race fixture, not a production
latency measurement. All 109 regression tests and pip check passed.

[Complete measurements](validation/two-core-blocks/summary.json) include
human-reference section scores, resource limits, caption comparisons and
transport-race evidence. Compressed raw reports preserve all native/paced
snapshots and transcripts. The earlier
[transcript differences](validation/delayed-playback/diff.html) remain valid
because the two-core transcript is identical to the four-core selection.
