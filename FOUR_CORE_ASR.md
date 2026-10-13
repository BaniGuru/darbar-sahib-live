# Rolling decoder comparison on four CPU cores and 8 GB RAM

Historical diagnostic runs before the delayed playback change. For current
deployment and the selected single-pass 4-second mode, see [DELAYED_PLAYBACK.md](DELAYED_PLAYBACK.md).
The 4 encoder-thread settings below describe those tests, not the new defaults.

Four-core/eight-GB validation retained the previous decoder's complete
transcript and every partial update. The original decoder is now available
for explicit live comparisons. The default remains optimized because the
original path built a large backlog without an accuracy gain on the test
recording. KenLM stays enabled; no model weights, beam width, language
weights or acoustic word confirmation rules were reduced.

## Deployment

Allocate four CPU cores and 8 GiB RAM in the deployment platform. From this
checkout, run one worker:

```bash
ENABLE_LANGUAGE_MODEL=True ASR_DECODER_MODE=optimized ASR_ENCODER_THREADS=4 \
  LIVE_STREAM_URL='https://live.sgpc.net:8442/;stream.mp3' \
  FFMPEG_PATH=/usr/bin/ffmpeg PORT=8000 \
  .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

Changing those process settings does not allocate hardware; provision the
CPU/memory budget separately. `asr_config.py` defaults to `decoder_mode =
"optimized"` and `enable_language_model = True`. Restart after changing them.
The environment can override the decoder mode. `/asr/health` must show
`decoder_mode=optimized`, `cpu_budget=4`, `encoder_threads=4`,
`decoder_beam_width=24`, `enable_language_model=true` and KenLM order 3,
weight 0.3, ngram mode. The corpus prior remains at weight 0.6.

Accuracy mode uses the original full-prefix scoring, original candidate
rescue bound and original content-keyed predictor interface. Its decoder
function's AST is identical to commit `ab2907c`, apart from its name. The
incremental scorer, additional per-label bounds and identity checkpoint
path introduced in `b2f3f2d` are bypassed in accuracy mode. Select
`ASR_DECODER_MODE=accuracy` to reproduce the original decoder, then restart.
This mode has no demonstrated accuracy advantage on the supplied inputs
and can be too slow for continuous live forwarding. Optimized mode remains
the recommended deployment setting.

The idle guard, 800 ms previews, four-second acoustic cache steps,
whole-word lookahead and eight-second pending-window cap remain in effect.

## Accuracy comparison

All three runs used enforced Docker cgroups: CPU quota `400000 100000`, CPU
affinity `0,1,2,3`, memory limit `8589934592` bytes and no swap. This tests
four cores/8 GiB on the available x86-64 host; it does not identify a
particular customer's processor. KenLM was enabled in every run.

Each decoder processed the original 52.0126875-second supplied example and
the first 1,200 seconds of the Bhai Agaykar Singh recording. The full
recording prefix contains 19,200,000 samples and 30,000 encoder frames.
All samples/frames were retained, including terminal audio. Every 800 ms
snapshot, committed/partial word timestamp and acoustic cache decision is
recorded, not just the final text.

| Decoder, 24 beams | WER | CER excluding spaces | Missing reference words |
| --- | ---: | ---: | ---: |
| legacy | 10.31% | 7.51% | 10 |
| optimized | 10.31% | 7.51% | 10 |
| accuracy | 10.31% | 7.51% | 10 |

The restored original mode matches the previous decoder at every update and in
the final result. The optimized implementation also matches every one of
the 1,500 recording updates and all 65 example updates, including partials,
commitments, word timestamps and cache decisions. There are no transcript
differences between these versions on the supplied inputs.

These results do not reproduce the reported live accuracy drop, and do not
demonstrate an accuracy gain from increasing CPU/memory. The original
decoder is available for comparison on the reported live case. A specific
incorrect phrase or recording remains necessary to diagnose that case.
Do not treat identical results on these recordings as proof about all live
input or the source/reconnect pipeline.

WER/CER use 572 human-reference words, ignoring punctuation, standalone
verse numbers and only terminal sihari/onkar. Internal vowels, repeated
words and word boundaries remain significant. Unannotated musical or
repetition gaps can contribute insertions; supplementary interval-coverage
scores are included in the JSON. Recognition does not read the reference.

See [word comparison](validation/four-core/diff.html),
[section metrics](validation/four-core/sections.csv) and
[complete summary](validation/four-core/summary.json). Compressed raw
reports retain every caption snapshot and full transcripts.

## Delivery latency

Final tests paced 100 ms PCM packets over the actual WebSocket API with
server and client inside the same four-core/8-GiB container. Each was
preceded by a complete HTTP control transcription. Within each test,
HTTP and WebSocket text, token IDs, word timings, sample/frame counts and
cache decisions agree. Both modes were tested with KenLM enabled. The
tests ran serially with no competing model. Native replay simulated a
40.28-second maximum backlog for the isolated original decoder, versus
8.23 seconds for optimized mode; averages below real time can still hide
sustained overload. In the live application, enough backlog can overflow
its short send queue and bounded replay history, dropping source audio.
Finite API parity does not guarantee the live forwarding pipeline never
skips audio.

| Recording | Median lag | p95 lag | Maximum lag |
| --- | ---: | ---: | ---: |
| Previous decoder: rapid 680–800 s | 1.551 s | 14.756 s | 16.238 s |
| Previous decoder: opening 200 s | 0.717 s | 6.129 s | 8.577 s |
| Optimized: rapid 680–800 s | 2.179 s | 9.409 s | 10.223 s |
| Optimized: opening 200 s | 0.589 s | 3.344 s | 4.707 s |

Lag is measured after the update's audio becomes available. These partial
delivery figures do not change the four-to-eight-second acoustic cache or
whole-word confirmation policy. The opening test includes the largest
burst seen in the initial baseline. Results depend on the processor and
other workloads; use the measured maxima rather than assuming a hard
two-second bound.

Initial legacy replay briefly overlapped regression checks during
startup/the example/early intro. Its simulated backlog is retained as
diagnostic evidence and is not a clean performance comparison. The
optimized and restored replay runs were isolated. Replay backlog is a
FIFO simulation; the table above reports actual paced delivery.

## Reproduction

The causal profiler is `scripts/benchmark_streaming_resources.py`. Supply
mono 16 kHz PCM16 WAV or float32 `.f32`; set the decoder mode explicitly and
run inside an actually capped container. The profiler records cgroups,
effective settings, update costs and simulated lag. Use the existing
`scripts/validate_streaming_asr.py` for paced API checks on a running server.
Stop the live decoder before benchmarking, and restart it afterward.

The verified image digest and package mounts documented in
[LOW_POWER_ASR.md](LOW_POWER_ASR.md) also apply here; change the Docker
limits to `--cpus=4 --cpuset-cpus=0,1,2,3 --memory=8g --memory-swap=8g`.
Choose CPU IDs allowed by the host. Retain TLS/artifact verification,
the unprivileged user and container safety options.
