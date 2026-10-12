# Offline model validation and long recordings

This branch starts from `main` (`e81db754638ca79706ec546b171795494ed9e53c`)
and continues to use its uncached `model/` ONNX weights. Long recordings
are decoded in bounded 30-second windows: 20-second cores with five seconds
of left and right audio context. Shared words reconcile overlaps; token
sequences are never cut at a nominal boundary. This bounds attention memory
and avoids independently filtering the same boundary word out of both windows.

The decoder retains four probability-pruned RNNT hypotheses. It uses a soft
word-context prior from the existing Gurbani database, then restores only
final short ੁ/ਿ using exact three-word corpus stem context and at least 80%
vote agreement. Restoration preserves word count and timestamps, and cannot
change consonants, other vowel marks, or word divisions. Raw ASR text
and tokens remain available alongside all correction records. The reference
transcript is used only for evaluation, never for decoding or restoration.

Defaults and controls:

- `ASR_ENCODER_THREADS=4`; decoder inference uses one CPU thread.
- `ASR_DECODER_BEAM_WIDTH=4`; set `1` for the original greedy decoder.
- `ASR_LANGUAGE_PRIOR_WEIGHT=0.6`; set `0` to disable the soft prior.
- `ASR_RESTORE_GURBANI_ORTHOGRAPHY=1`; set `0` for raw ASR spelling.

Existing live-window behavior remains 750 ms refreshes with six-to-ten-second
rolling windows. The 30-second window policy applies to finite long-file
transcription and authoritative capture flushes; this is not a cache-aware
encoder or a guarantee of streaming-model latency.

## Validated comparison

The first exactly 1,200 seconds of the supplied Hugging Face recording were
converted once to a 16 kHz mono PCM WAV and sent to both models. Its 99
complete human-reference sections contain 572 words. All 19,200,000 samples
were covered by 60 contiguous ownership intervals; encoder windows never
exceeded 30 seconds. The first five minutes were used for pilots.

| Path | WER | Missing words |
| --- | ---: | ---: |
| Original main greedy, same overlap reconciliation | 39.51% | 40 |
| Updated offline raw ASR | 35.14% | 14 |
| Updated offline displayed transcript | **16.78%** | **14** |
| Current streaming pipeline | 18.36% | 25 |
| Streaming with the same spelling restoration, analysis only | 17.48% | 25 |

The updated offline pipeline has fewer missing words and lower displayed
word error rate on this sample. Its raw decoder has higher WER than streaming;
canonical spelling restoration accounts for much of the improvement. Offline
also uses future audio unavailable to a low-latency stream. These results do
not establish general model superiority or equal live-partial accuracy.
The matched spelling-control row does not modify the streaming branch.

[Comparison report](validation/offline-vs-streaming20m/README.md),
[highlighted word diff](validation/offline-vs-streaming20m/diff.html), and
[99-section CSV](validation/offline-vs-streaming20m/sections.csv) include
complete human/reference hypotheses, raw outputs, exact error accounting,
input hashes, corpus correction records, and source provenance.

## Reproduce

From the existing checkout, use one Uvicorn worker:

```sh
FFMPEG_PATH=/usr/bin/ffmpeg PORT=8000 .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python scripts/evaluate_offline_asr.py --audio first20m.wav --reference reference20m.json --report /tmp/offline20m.json
```

In the current environment, the input and reference are in
`/workspace/offline-model-validation/`. The complete source recording and
human transcript remain in `/workspace/verified-kirtan-validation/`.
The input WAV was generated with:

```sh
ffmpeg -i recording.mp3 -t 1200 -ar 16000 -ac 1 -c:a pcm_s16le first20m.wav
```

To reproduce the greedy control, archive `main:server_asr_offline.py` and
pass that file through `--source`; the evaluator uses identical bounded
windows and a controlled CPU budget. `--stitch ownership` reproduces the
initial timestamp-only adapter; the default is shared-word anchors. Use
`--hypothesis` to rescore an existing report without another model run.

All 16 regression checks passed: boundary-word preservation despite moving
timestamps, real repetitions, bounded attention and terminal-sample coverage,
RNNT blank state, local-ID feedback, later acoustic promotion, constrained
spelling restoration, and exact scoring of missing/split words and vowel marks.
The actual HTTP long-file path also passed on the original 52-second supplied
WAV, retaining 832,203 samples and exposing raw/corrected transcript metadata.

The original offline full-file entry point has no attention-memory bound.
Its baseline therefore uses the same bounded adapter rather than attempting
an unsafe hour-long single encoder call. Elapsed timings from comparison jobs
that ran concurrently are not a controlled model speed comparison.

After the model-bearing commit was pushed and fetched back, the actual HTTP
endpoint transcribed the complete 20-minute input with default CPU settings.
Its text exactly matched the pre-push run, and all 19,200,000 samples were
retained. WER remained 16.78%, with 70 substitutions, 14 deletions and 12
insertions. The isolated HTTP run took 112.15 seconds. The post-push JSON
verification is linked from the comparison report.

The complete app was also restarted with the documented FFmpeg/one-worker
command. Its five web/health/state endpoints, live source connection, ASR
connection and five changing-caption messages were verified. This readiness
check measures delivery and startup, not accuracy of the current live
passage, which has no human reference.
