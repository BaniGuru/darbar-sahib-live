# Streaming on two CPU cores and 4 GB RAM

These are historical rolling-mode measurements. The current deployment uses
[30-second delayed playback and single-pass 4-second blocks](DELAYED_PLAYBACK.md)
on four cores / 8 GiB, with KenLM enabled. Use `ASR_RECOGNITION_MODE=rolling`
to reproduce the approach measured below.

These measurements describe the low-power decoder introduced in `b2f3f2d`.
It remains the default `ASR_DECODER_MODE=optimized`. An explicit
`ASR_DECODER_MODE=accuracy` restores pre-optimization full-prefix scoring and
predictor handling for comparison on reports of worse live partials and
committed words. Keep KenLM enabled. See [FOUR_CORE_ASR.md](FOUR_CORE_ASR.md)
for the four-core/eight-GB comparison; the two-core timings below apply to
optimized mode, not the restored original decoder.

The test used a Docker container with two pinned cores, a two-core CPU quota,
4 GiB RAM, no swap and one ASR worker. This constrains the service on the
current x86-64 host; it is not a measurement of a particular customer's CPU.
See [validation/low-power/summary.json](validation/low-power/summary.json).

The default four-thread encoder could exceed the available CPU budget. It
spent 65 of 68 seconds encoding the 52-second kirtan example; direct KenLM
scoring took only 0.02 seconds. Its simulated FIFO backlog reached 18 seconds.
Even with two encoder threads, the 24-hypothesis decoder built a 16-second
backlog on the two-minute rapid passage. It repeatedly decoded candidate
text prefixes and copied/serialized predictor states across blank frames.

The fix detects CPU affinity and cgroup quotas when selecting the default
encoder thread count. Two-core deployments use two encoder threads. The
24-hypothesis beam and acoustic/LM weights remain unchanged. Word scoring now
advances incrementally and uses conservative per-token rescue bounds to skip
candidates that cannot survive the original score threshold. Numerals,
punctuation and unknown pieces retain the original full-prefix fallback.
Immutable predictor checkpoints use a bounded cache with strong references,
so their arrays need not be repeatedly concatenated and serialized. Mutable
inputs retain the original content-keyed interface.

These changes preserve the original KenLM-on transcript, token IDs, word
timestamps and cache decisions for the tested normal and rapid recordings.
The test suite also compares next-token scores and pruning bounds with the
original scorers, and predictor logits/states with the exported ONNX graph.

## Language-model control

Edit `asr_config.py`:

```python
enable_language_model = True   # Default: load and use the KenLM n-gram.
# enable_language_model = False
```

Or override it at startup:

```bash
ENABLE_LANGUAGE_MODEL=False FFMPEG_PATH=/usr/bin/ffmpeg PORT=8000 \
  .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

Restart after changing the setting. `False` disables the optional KenLM
binary/import/scoring. The existing Gurbani corpus word prior stays enabled;
removing it lost many more words in the diagnostic run. Set
`ASR_LANGUAGE_PRIOR_WEIGHT=0` only if intentionally disabling that prior too.
`/asr/health` reports `enable_language_model`, `cpu_budget`, effective threads,
beam width and both prior weights. Explicit `ASR_ENCODER_THREADS` and
`ASR_BEAM_WIDTH` override the defaults.

Keep KenLM enabled when accuracy matters, including on a two-core deployment.
Disabling it alone does not remove the main bottleneck; the threading and
cache/scoring fixes do that. The switch remains available for deployments
that prefer its modest speed improvement or do not have KenLM installed.

## Measured results

The two-minute stress recording is seconds 680–800 of the supplied Bhai
Agaykar Singh recording. Final tests used actual 100 ms paced WebSocket
packets after an HTTP control transcription, with both client and server
inside the same two-core/4-GiB container. No other model ran during these
paced tests. Every sample and encoder frame survived; HTTP and WebSocket
word timings, tokens and cache decisions match within each mode.

| Final mode, 24 hypotheses | Median delivery lag | p95 lag | Maximum lag | WER | CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| KenLM on | 0.555 s | 1.938 s | 2.699 s | 15.12% | 13.10% |
| KenLM off, corpus prior on | 0.538 s | 1.672 s | 2.598 s | 18.02% | 16.21% |

WER/CER use the 172 reference words in fully contained human-verified
sections of this clip, ignoring only terminal sihari/onkar in addition to
punctuation and standalone verse numbers. CER excludes spaces. Unverified
edge audio can contribute insertions; the JSON includes supplementary
interval-coverage scoring. These are clip rates, not whole-recording rates.
KenLM-on has five reference deletions; disabling KenLM has nine.

The target of 1–2 seconds is met for 95% of updates in this stress test;
brief bursts can still exceed two seconds. This measures delivery after an
update's audio becomes available. The existing four-to-eight-second acoustic
cache/whole-word confirmation policy remains unchanged. Partial previews
still refresh every 800 ms.

Peak cgroup memory during the paced runs was about 1.37–1.46 GiB. A separate
combined-model/pankti-index check loaded all 130,840 panktis and replayed the
actual ASR messages: peak process RSS was 1.53 GiB; matching cost was 2 ms
median, 163 ms p95 and 194 ms maximum. No memory-limit or OOM events occurred.
This combined check measures matcher cost separately from ASR inference;
it is not an additional end-to-end paced measurement.

Rejected tuning is retained in the report. A one-thread encoder was slower.
Eight hypotheses were fast but raised rapid-passage WER to 27.33%. Disabling
both language priors raised WER to 34.30%. Turning off ORT thread spinning
also slowed this workload. These settings are not the defaults.

## Reproduction

The prepared image is `python:3.12-slim` at digest
`sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`.
It uses the checkout's installed Python 3.12 packages through `PYTHONPATH`.
Choose two CPU IDs permitted by the host affinity; this instance uses 0,1.

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --network none --cpus=2 --cpuset-cpus=0,1 --memory=4g --memory-swap=4g \
  --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=256 \
  --mount type=bind,src=/workspace/darbar-sahib-live,dst=/project,readonly \
  --mount type=bind,src=/workspace/stream-missing-validation/rapid.wav,dst=/audio.wav,readonly \
  --mount type=bind,src=/workspace/low-power-validation,dst=/results \
  -e PYTHONPATH=/project/.venv/lib/python3.12/site-packages:/project \
  -e ENABLE_LANGUAGE_MODEL=True \
  python:3.12-slim python /project/scripts/benchmark_streaming_resources.py \
  --audio /audio.wav --output /results/replay.json
```

The causal profiler records enforced cgroup limits, component costs,
transcripts and a clearly labeled FIFO-lag simulation. The compressed result
files retain the actual paced tests, full transcripts and diagnostic traces.
Use the existing `scripts/validate_streaming_asr.py` for paced API validation
on a running service; that helper needs FFmpeg for input decoding.
The 88 regression tests pass with
`.venv/bin/python -m unittest discover -s tests`.
