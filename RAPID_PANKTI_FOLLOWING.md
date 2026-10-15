The paired display described below is historical (commit `41093d3`). The
current player shows **one pankti at a time**, switching at its prepared
first-word boundary between polls. See [SINGLE_PANKTI_DISPLAY.md](SINGLE_PANKTI_DISPLAY.md).
The matching improvements and recording fixtures below remain in use.

The live server prepares pankti IDs against ASR word timestamps once.
Each listener's one-second state request selects the newest prepared event
at that listener's delayed audio position. It performs neither Gurbani
matching nor line grouping. `/api/live/state.playback_words` includes the
word's `pankti_id`, `shabad_id`, optional bani position, and `audio_timestamp`;
`display_panktis` contains the ready-to-render canonical text/translations.
The existing scalar pankti fields remain compatible.

A distinctive, ordered two-word match within the selected shabad advances
immediately, including a sung backward repeat. Ambiguous shared pairs keep
confirmation/context safeguards. Every newly received timed word in a batch
is considered in order, so fast intermediate lines get their own source-time
boundary. Unchanged ASR snapshots reuse the result; global shabad lookup
runs once per changed batch rather than once per word. Outside-shabad
confirmation still counts separate ASR updates. Acoustic recognition,
KenLM, four-second blocks and ten-second playback delay are unchanged.

The recording replay identified and fixed two timestamp problems. Unmatched
trailing words no longer make an old alignment appear current. A bani line
sharing middle words with its predecessor cannot be backdated into the
predecessor's phrase. Very short bani lines are recorded as heard before
recording their auto-next prediction. Superseded renewals of an older line
cannot sort after a confirmed correction and hide it. A repeated refrain's
end/start pair does not jump into another pankti's middle.

For close consecutive lines the server prepares a two-line display, in
reading order, around the transition. Only heard lines establish a pair;
an auto-next prediction alone cannot. The latest supporting word in the
first line must be within six seconds of the next line's start. The pair
is shown within six seconds on either side of that start. This handles
held notes without grouping lines separated by long gaps. The canonical
next-pankti map must agree within a shabad; bani occurrences must also be
consecutive on the same route. Skipped lines, backward jumps (including a
bani route restarting a shabad), unrelated shabads and greetings break the
pair. At most two lines appear. A backward jump shows its latest line alone.

The date and India clock use 18px as requested, with “ਅੰਮ੍ਰਿਤਸਰ” before India
time. The player clock reads “Start Listening” while stopped, and shows the
listener's local time during playback or the post-greeting wait. Metadata
wraps on small phones, and paired verses keep their full text visible.
Frontend cache version is 17. No runtime dependencies were added.

Validation uses the exact public recording supplied in the request:

- File positions **02:35:00–02:40:00**, Asa Ki Vaar: 300 seconds.
- File positions **12:08:36–12:11:00**, fast singing: 144 seconds.
- Each capture includes 16 seconds before and 12 seconds after the requested
  range, using the production block recognizer with KenLM enabled. Total
  acoustic input is 328 and 172 seconds. Audio was transcribed once per
  capture; the same causal messages feed both matcher implementations.
- The source MP3 has 73,799.549375 seconds, 590,396,395 bytes, SHA-256
  `23d8c9f385f9ecbea94c52d8765895b04cc859cb1d30f43092dd36553851f729`.
- Baseline is commit `f48d81a`. Listener state was sampled once per second,
  ten seconds behind received input: 301 and 145 samples, with no future
  scalar pankti timestamps. The requested HYN0/FNN1 pair is present in the
  fast recording's prepared display.

| ASR-to-pankti timing check | Baseline | Updated |
| --- | ---: | ---: |
| Asa Ki Vaar, unambiguous exact word pairs | 179/202 | 202/202 |
| Fast passage, unambiguous exact word pairs | 32/47 | 46/47 |

These checks measure following the recognized words within shabads both
implementations selected. They are **not WER, CER or human transcript
accuracy**. Three cyclic refrain-boundary pairs are explicitly excluded and
listed, because “ਨਾਮ ਜਪਹੁ” across a repetition does not prove that the line
“ਨਾਨਕ ਨਾਮੁ ਜਪਹੁ ਗੁਨਤਾਸੁ” was sung. The remaining fast-passage discrepancy is
also listed at the end of the requested range; the diagnostic interprets
another cross-boundary “ਨਾਮ ਜਪਹੁ” pair as that line. ASR spelling mistakes
and initial shabad ambiguity remain possible. No human reference was
provided for this recording, and no acoustic accuracy claim is made.

All 131 Python tests pass, including fast batches, two-word forward/backward
switches, true alignment endpoints, refrain wrap protection, short bani
lines, word IDs, prepared-poll reuse, grouping expiry and backward/gap
exclusions. Chromium checks at 1280×800, 390×844 and 320×700 cover the actual
HYN0/FNN1 text, paired/single transitions, greeting/recovery, clock labels,
18px metadata and responsive layout. Real live AAC play/stop and listener
count checks also pass. See `validation/rapid-pankti` for full timelines,
word boundaries and performance measurements on the cloud CPU. Those
measurements are not a latency guarantee for every production processor.

Reproduce from the checkout, with the source MP3 downloaded outside Git:

```bash
ENABLE_LANGUAGE_MODEL=True .venv/bin/python scripts/capture_stream_asr.py \
  --audio /path/to/recording.mp3 --start 9284 --seconds 328 --output /tmp/asa
ENABLE_LANGUAGE_MODEL=True .venv/bin/python scripts/capture_stream_asr.py \
  --audio /path/to/recording.mp3 --start 43700 --seconds 172 --output /tmp/rapid
.venv/bin/python scripts/validate_pankti_timing.py \
  --events /tmp/asa.jsonl.gz --asr-report /tmp/asa.json --seconds 300 --report /tmp/asa-match.json
.venv/bin/python scripts/validate_pankti_timing.py \
  --events /tmp/rapid.jsonl.gz --asr-report /tmp/rapid.json --seconds 144 --report /tmp/rapid-match.json
.venv/bin/python -m unittest discover -s tests
```

The replay defaults to the same baseline above and a 16-second warm-up.
Small compressed ASR event fixtures and capture summaries are retained in
`validation/rapid-pankti` so matcher tests can run without downloading the
590 MB MP3. Native PCM and the source recording are outside Git. Optional
browser checks use `scripts/validate_player_ui.cjs` with external Playwright
as documented in `PLAYER_UI.md`; `PLAYER_UI_PAIR_FIXTURE` can point to the
canonical two-line fixture in the validation directory.
