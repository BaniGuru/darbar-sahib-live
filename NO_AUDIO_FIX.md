# Idle-stream hallucination fix

Branch `no-audio-fix` starts from main `d718a7ec7c0a4e9d34a54d0c2eb11ad212eeac23`.

The SGPC 8442 source delivers AAC even before kirtan starts. Its idle signal is
low-level broadband noise, not digital silence. Feature normalization raises
that noise into the acoustic model's working range; the previous decoder
invented Gurbani words from it.

`audio_activity.py` examines raw 16 kHz PCM before normalization, in 20 ms
frames with DC removed. Audio above 0.001 RMS (−60 dBFS) remains eligible.
Quieter tonal audio also remains eligible when spectral flatness over
100–4000 Hz is below 0.15. Exact silence/DC and quiet broadband noise are
inactive. This deliberately preserves uncertain louder audio and very soft
singing; it does not attempt to suppress every possible kind of noise.

`StreamingRecognizer` suppresses RNNT decoding only when the entire pending
window is inactive and no credible audio was heard in the previous four
seconds. This preserves pauses and word endings. Every PCM sample still
reaches the existing frontend/encoder, and encoder caches, timestamps, 800 ms
refreshes, four-second cache steps and eight-second pending limit continue
normally. If the stream began idle, its acoustic cache resets once when the
first credible audio arrives: normalized startup noise must not become
phonetic context for the first words. Raw PCM, history, predictor and absolute
clocks remain intact. Later pauses never trigger this reset.
Existing predictor hypotheses and genuine last words survive idle
periods and EOF. New audio immediately uses the existing preview schedule.
No volume normalization, acoustic model, beam width, KenLM weight or boundary
word policy changes were made.

ASR messages add `audio_active`. The live matcher does not refresh a previous
pankti from an inactive update; previously recognized text remains visible in
the transcript. Clients without the field retain their existing behavior.
ASR health reports `silence_guard` and `silence_hangover_seconds`.
`ASR_SILENCE_GUARD=0` disables the guard for a diagnostic baseline.

## Validation

The captured source is `https://live.sgpc.net:8442/`, content type `audio/aacp`.
The AAC SHA256 is
`b674e2ee5a132325efd5a26f4319e6da1e1ba23338175843d251d9dd037835d4`.
It decodes to 1,046,198 samples / 65.387375 seconds with RMS 0.000469762
(approximately −66.6 dBFS). The previous recognizer emitted “ਸੁਹੇਲੇ”.
With the guard every partial and the final transcript are empty.

Recording comparisons and processing times are saved in
[validation/no-audio/results.json](validation/no-audio/results.json).
They compare the same causal decoder with the guard enabled/disabled, including
all previews, final tokens, word timings, cache events and sample/frame counts.
The attenuated recording uses a 0.01 amplitude multiplier (−40 dB); this is
an accuracy parity check against the same quiet input, not an assertion that
attenuation itself leaves recognition unchanged. CPU replay timings include
preview recomputation and are unpaced. These samples establish regression
parity for the tested audio, not universal speech accuracy or a new human WER.

The tests include noise suppression, genuine quiet tones and short tails,
onsets after idle, hangover, packet-size parity, no stale match renewal, and a
two-hour idle simulation with the acoustic encoder mocked. The simulation
checks clocks and memory bounds, not two hours of real live recognition.
Higher-order KenLM tests use small real ARPA fixtures because main now ships
only the selected 3-gram binary.

Reproduce with an idle capture and an audible recording:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/validate_no_audio_guard.py \
  --idle /path/to/idle.aac --audio /path/to/kirtan.mp3 \
  --report /tmp/no-audio.json
```

For paced HTTP/WebSocket parity, run `server_asr_stream:app` on port 8001,
then use the existing `scripts/validate_streaming_asr.py` with `--packet-ms 100`.
`validation/no-audio/paced.json` records the actual paced check for this change.
EOF must still flush all samples and preserve committed word prefixes.

Final checks: 78 regressions pass. In a 32-second fixture containing eight
seconds of captured idle noise, twelve seconds of kirtan, then twelve seconds
of idle noise, HTTP and paced WebSocket results agree. The opening phrase
"ਕਰਿ ਕਿਰਪਾ ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ ਤੇਰੀ ਓਟ ਪੂਰਨ" matches the clean recording
control. No word is timestamped before the real audio; the initial idle
previews are blank and the idle suffix is marked inactive. Paced delivery lag
is 413 ms median, 560 ms p95 and 612 ms maximum on this machine.

During the final full-application check the SGPC source began carrying
audible audio (fresh capture RMS 0.104, approximately −19.7 dBFS), so live
readiness checks require connected source/ASR, changing captions and a pankti.
They do not claim that the current live source is still idle. Use the explicit
`https://live.sgpc.net:8442/;stream.mp3` audio URL: with the app browser User-Agent,
the root URL redirects to the Shoutcast administration page.
