# Delayed playback and single-pass recognition

Listeners hear the original AAC/MP3 stream 30 seconds behind its received
media edge. Recognition and Gurbani matching run ahead of playback. Pankti
and debug captions use each listener's audio clock, so future matches are
not shown while earlier audio is playing. New listeners join near the
delayed edge; a cold server buffers first. Normal browser/network buffering
can add further delay.

## Deployment

Provision four CPU cores and 8 GiB RAM, then run one worker:

```bash
ENABLE_LANGUAGE_MODEL=True PLAYBACK_DELAY_SECONDS=30 \
  ASR_RECOGNITION_MODE=blocks ASR_BLOCK_SECONDS=4 \
  LIVE_STREAM_URL='https://live.sgpc.net:8442/;stream.mp3' \
  FFMPEG_PATH=/usr/bin/ffmpeg PORT=8000 \
  .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

These settings are defaults except the explicitly selected source URL.
KenLM 3-gram remains enabled, with 24 beams, weight 0.3, ngram scoring and
zero word bonus. The corpus prior remains 0.6. Four-core containers default
to two encoder threads and one BLAS thread, leaving capacity for decoding,
feature extraction and matching. `threadpoolctl` applies the BLAS limit even
when NumPy was imported by the application first. CPU/memory allocation is
controlled by the deployment platform, not these environment variables.

`/asr/health` exposes recognition mode, block size, encoder/BLAS thread
counts and actual BLAS pools. `/health` exposes playback delay and compressed
audio buffer duration. `ASR_RECOGNITION_MODE=rolling` retains the previous
800 ms rolling approach for diagnostics; `ASR_DECODER_MODE=accuracy` selects
the pre-optimization decoder for comparison. Neither is recommended for
this delayed deployment. Restart after changing process settings.

## Recognition and synchronization

The NeMo [202,100] model still uses 40 ms encoder frames. Each 4-second
acoustic block is encoded and decoded once; repeated rolling previews are
removed. Four seconds of future PCM informs frontend normalization before
a block is processed. The pending PCM window is at most eight seconds,
plus bounded 16-second normalization history. The required eight frontend
context frames are retained; this is not another pass over the acoustic
block. Encoder calls do not finalize words.

Predictor state, competing unfinished word tokens and KenLM context survive
cache advances. A word starting at 3.75 seconds can finish in a later block
without being divided at four seconds. Lexical prefixes settle after a
shared next-word boundary plus one second of acoustic evidence. Long sung
words or instrumental gaps can postpone commitment beyond 30 seconds;
timed provisional hypotheses remain usable for search and playback debug
captions. This avoids forcing a trailing word to end merely because a cache
block ended. An explicit WebSocket `finish` drains all terminal samples.

Compressed frames are parsed incrementally, timestamped by actual frame
sample counts and buffered without transcoding or altering frame bytes.
The same frame sequence feeds FFmpeg/ASR immediately. The audio buffer
drains during an upstream outage. A listener reconnect resets its media
anchor and closes its prior queue. A slow listener disconnects rather than
silently losing audio. The ASR send queue now absorbs up to 38 seconds by
default, rather than dropping a source startup burst after two seconds;
the recognizer's eight-second pending-window cap is separate.

Only actual block progress is sent as matching evidence. Confirmed pankti
events can be backdated to the first supported two-word phrase, so the
listener sees the line as that phrase begins. Existing shabad switching,
Asa Ki Vaar interleaves and ordered bani following remain in use.

## Completed accuracy comparison

Both candidates transcribed the original 52.0126875-second example and the
first 1,200 seconds of the Bhai Agaykar Singh recording. Each native replay
ran serially with an enforced four-core quota/affinity, 8 GiB limit and no
swap on the available x86-64 host. All 19,200,000 samples and 30,000 encoder
frames were retained. KenLM was enabled for every run.

| Approach | WER | CER excluding spaces | Deletions | Runtime for 20 min |
| --- | ---: | ---: | ---: | ---: |
| Previous rolling | 10.31% | 7.51% | 10 | 778.05 s |
| Single-pass 2 s | 15.73% | 12.32% | 23 | 175.02 s |
| **Single-pass 4 s** | **10.14%** | **7.47%** | **10** | **122.00 s** |

Selected 4 seconds: accuracy improves slightly, with no increase in missing
reference words. This does not establish that the reported live accuracy
regression is fully fixed. The original and optimized rolling decoders
produced identical snapshots on these supplied inputs; see
[FOUR_CORE_ASR.md](FOUR_CORE_ASR.md). The runtime comparison includes the
thread-budget change as well as removing repeated acoustic work, and is
not an isolated measurement of either factor.

Scoring uses 572 human-reference words. Only terminal sihari/onkar,
punctuation and standalone verse numbers are ignored. Internal vowels,
word boundaries and repetitions count. Unannotated musical/repetition gaps
can contribute insertions; interval-only scores are supplementary.
The 4-second run used 300 encoder calls, with zero acoustic re-encodes.
Actual active-block computation cost was 0.621 s at p95, 1.387 s maximum;
collection-only ticks are excluded. Its real-time factor was 0.102.

A separate serial paced API test used 100 ms packets on the rapid 680–800 s
passage with server/client inside the same four-core/8-GiB container. HTTP
and WebSocket text, tokens, word times, sample/frame counts and cache events
agreed. Recognition delivery lag from the processed audio frontier was
4.434 s median, 4.875 s p95 and 4.932 s maximum. This includes deliberate
4-second normalization lookahead and excludes the 30-second listener delay.
Those settings leave about 25 seconds for matching before the listener
hears processed audio; they are not a guarantee for every CPU or live word.

See [word-by-word differences](validation/delayed-playback/diff.html),
[section scores](validation/delayed-playback/sections.csv) and
[full metrics](validation/delayed-playback/summary.json). Compressed raw
reports retain every transcript snapshot and final result. AAC and MP3
frame parsing reproduced the original decoded PCM exactly on the captured
AAC file and the first 45 seconds of the complete MP3.

Full live readiness additionally passed the SGPC 8442 source, ASR, public
AAC endpoint and delayed state queries. A 51.223-second public capture
contained only complete original frames; its parsed and decoded durations
agree within one 16 kHz sample. All 50 sampled states displayed a matched
pankti at or before the listener playhead, and reconnect advanced the anchor.
This live check validates delivery and synchronization, not human-scored
accuracy. See [live evidence](validation/delayed-playback/live-ready.json).

The native 65.387-second idle capture produces no text in block mode. On
the example attenuated by 40 dB, enabling/disabling the idle guard produces
identical tokens and word timings. Attenuation itself changes one word
relative to the normal-volume example; this remains a model/frontend
limitation, so volume invariance is not claimed. See
[guard evidence](validation/delayed-playback/guard-validation.json).

Run regression checks:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
```

For reproduction, benchmark default block mode with
`scripts/benchmark_streaming_resources.py`, compare paced HTTP/WS with
`scripts/validate_streaming_asr.py`, and score the full result with
`scripts/evaluate_reference_asr.py --ignore-final-short-vowels`.
Do not benchmark alongside another live decoder. Source recordings and
reference hashes are in the reports; large audio files are not committed.
