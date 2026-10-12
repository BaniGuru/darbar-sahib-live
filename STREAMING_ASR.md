# Cache-aware streaming ASR

The NeMo ONNX model uses `[202, 100]` attention context and 40 ms encoder
frames. Encoder caches advance in four-second steps after adaptive word
lookahead. Previews re-decode the pending audio every 800 ms without changing
the retained caches. The uncached audio window never exceeds eight seconds.

The exported encoder discards two pre-encoded frames. Each full invocation
therefore receives eight preceding frontend frames plus 400 new frontend
frames, producing all 100 encoder frames. Frontend processing includes up to
16 seconds of preceding audio for continuous pre-emphasis/STFT and less
variable feature normalization. Audio packets crossing a cache boundary retain
their overflow samples.

The decoder keeps up to 24 probability-pruned RNNT hypotheses with their predictor
states across cache updates. All unresolved text remains visible in
`partial_text`, including competing spellings or word divisions. Later audio
can replace a prior block's word choice, rather than only append characters
to an irreversible greedy choice.

The decoder also uses a soft word-context prior built from the existing
`data/bani.db` Gurbani corpus, independently of evaluation transcripts.
It compares contextual word-prefix probabilities with corpus-wide prefix
probabilities, backs off for sparse contexts, and bounds each contribution.
The prior reranks acoustic alternatives while allowing unseen words and
ordinary speech. It carries the last settled word across cache updates.
`ASR_LANGUAGE_PRIOR_WEIGHT` defaults to `0.6`; set it to `0` for an acoustic
control run. `ASR_BEAM_WIDTH` defaults to `24`; set it to `4` for the previous
search width. The unchanged ONNX decoder is factored into encoder projection,
predictor and joint computations. A bounded 16,384-entry cache reuses predictor
outputs only for identical labels and complete recurrent input states; blank
frames do not rerun the LSTM. Acoustic projections have a separate 128-entry
cache. Both caches include input shape, dtype and bytes in their keys. Joint
scores still use each current acoustic frame. Set `ASR_CACHE_PREDICTOR=0` for
the original combined decoder graph. Corpus prefix scores have a bounded
16,384-entry cache, with a separate 32,768-entry token-prefix cache. Uncached
predictor states are evaluated in a single batch. Sibling label hypotheses share immutable predictor
arrays. These changes make the wider search practical without changing model
weights, acoustic scores or the audio window limits. Settled word records and text are
cached for display, so previews process only the mutable suffix between
cache promotions.

At the four-second boundary, the recognizer keeps the audio uncached while
the boundary word finishes. The following word's agreed start (within 120 ms
across retained hypotheses) provides a conservative end marker. After one
more second of audio, the entire boundary word settles and only the first
four seconds enter the retained encoder cache. Lookahead audio stays pending
for the next window. Separate encoder and predictor positions let the next
decode skip the settled word's frames without losing or duplicating its tokens.

For example, a word starting near 3.75 seconds and ending at 6.0 seconds
remains revisable at 6.4 seconds. It settles at the next 800 ms refresh after
7.0 seconds, which is 7.2 seconds; the cache then advances to 4.0 seconds,
leaving 3.2 seconds of lookahead pending. This behavior is covered by a
controlled decoder test.

At eight seconds the cache must advance by four seconds even if the word's
end is still unconfirmed. Its hypotheses remain partial and continue with
the retained lookahead; the limit does not declare the word complete. The
exported graph is evaluated in provisional blocks of at most four seconds,
so its output-frame cap cannot discard the second half of an extended window.
Previews stay visible while a cache decision waits.

`cache_committed_until` reports the acoustic cache clock;
`text_committed_until` reports the settled word clock. `committed_until` and
`text_commit_seconds` retain their legacy cache meanings for compatibility.

For finite recordings, send `{"type":"finish"}` after the last binary PCM
packet and wait for the `final` response before closing the WebSocket. This
flushes audio shorter than the 800 ms update interval and commits the final word.
The PCM format is 16 kHz mono float32 little-endian.

The encoder uses up to four CPU threads by default. Set `ASR_ENCODER_THREADS`
to adjust this for the host; use one Uvicorn worker.

## Validation

From the repository directory:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/uvicorn server_asr_stream:app --host 127.0.0.1 --port 8001 --workers 1
# In another terminal, with the server already running:
.venv/bin/python scripts/validate_streaming_asr.py test_audios_darbar_sahib_kirtan.mp3 --url http://127.0.0.1:8001
# Score an entire recording against human-verified timed sections:
.venv/bin/python scripts/evaluate_reference_asr.py --audio recording.mp3 --reference aligned_transcript.json --report /tmp/verified-report.json
```

Human-reference evaluation replays the complete recording causally in
800 ms packets. It runs faster than real time by default; add `--paced` to
use the recording's wall clock. Reference text is supplied only to the
scorer, never to the recognizer. JSON sections may use `transcript`,
`start_time`, and `end_time`, or `text`, `start`, and `end`.
The report includes exact word substitutions, deletions, insertions,
character error rate, section alignments, processing times, and the complete
recognized text and timed words. Normalization preserves vowel marks and
word divisions, so a false `ਵਾਰ ਗੁਰੂ` split still counts as an error.
The primary score includes every recognized word. A supplementary score
selects words starting inside verified intervals with a declared 500 ms
tolerance and reports words outside those intervals separately.

For the complete application, use `--url http://127.0.0.1:8000/asr` instead.
The validation script transcribes the entire file over HTTP, then sends the
same recording over WebSocket in real time using 100 ms packets. Both use
the streaming model's cached decoding path; this checks streaming parity,
not accuracy against a human-labeled transcript or a different offline model.

The following supplied-MP3 and 87.5-second live results were recorded before
the word-context prior. Updated results appear below.

Results for the supplied MP3 (SHA-256
`90492e044565751b7b91f22767e51aa7bc494c4922e63a2c6a3343c1bc143ac6`):

| Check | Result |
| --- | --- |
| Recording length | 52.0126875 seconds, 832,203 samples |
| Full-file versus live final transcript | Identical text and token IDs |
| Encoder coverage | All 1,301 frames retained |
| Preview cadence | 65 updates at 800 ms intervals |
| Cache promotion | 12 deferred four-second steps, followed by EOF flush |
| Peak uncached window | 8.0 seconds |
| Delay after an update's audio became available | Median 420 ms; p95 588 ms; maximum 787 ms |
| First nonempty hypothesis | Audio position 1.6 s, received at 1.808 s |
| Full-file inference | 21.461 seconds on the validation host |
| Regression suite | Eighteen tests passed |

At 4.0 seconds the visible trailing fragment is `ਪ੍ਰ`; at 4.8 seconds it
continues as `ਪ੍ਰਭ ਦੀਨ`, without a space inside `ਪ੍ਰਭ`. The last `ਗੋਪਾਲਾ`
remains visible at 52.0 seconds and is retained in the final response.

Final transcript:

> ਕਰਿ ਕਿਰਪਾ ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ ਤੇਰੀ ਓਟ ਪੂਰਨ ਗੋਪਾਲ ਕਰਿ ਕਿਰਪਾ ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ ਤੇਰੀ ਓਟ ਪੂਰਨ ਗੋਪਾਲਾ ਕਰਿ ਕਿਰਪਾ ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ ਤੇਰੀ ਕੀ ਓਟ ਪੂਰਨ ਗੋਪਾਲਾ ਤੇਰੀ ਓਟ ਪੂਰਨ ਗੋਪਾਲਾ

The phrase variation `ਤੇਰੀ ਕੀ ਓਟ` is present in both full-file and streaming
output. A verified reference transcript is needed to quantify word error rate.
Latency measurements describe this CPU host and exclude network transit.

## Live word-boundary validation

A fresh recording was captured from `https://live.sgpc.net:8443/;stream.mp3`
and compared using full-file inference and paced WebSocket replay. The test
also checks that all samples and frames survive, text fields and word records
agree, and 800 ms updates continue while four-second cache steps wait for
word lookahead. Both packetization and full-file inference must make the same
cache decisions and respect the eight-second limit. The model still makes
recognition mistakes; matching those two paths
does not establish a human-reference word error rate.

| Live recording check | Result |
| --- | --- |
| MP3 SHA-256 | `08a062bda6bd888a5b727d07a7afe7ffabdf85b257da2d1d1867a82957195592` |
| Audio length | 87.53925 seconds, 1,400,628 samples |
| Encoder coverage | All 2,189 frames retained |
| Full-file versus live final transcript | Identical text and token IDs |
| Preview cadence / acoustic cache commits | 109 updates at 800 ms / 21 deferred four-second steps |
| Peak uncached window | 8.0 seconds |
| Delay after update audio became available | Median 411 ms; p95 575 ms; maximum 681 ms |
| First nonempty hypothesis | Received at 1.768 s for audio position 1.6 s |
| Full-file inference | 38.312 seconds on the validation host |

The supplied recording and this live recording both stayed ahead of the
800 ms update cadence, without an accumulating inference backlog. All settled
words remained present in the final transcript.

Regression tests additionally cover a visible `ਵਾਰ` interpretation being
replaced by `ਵਾਹਿਗੁਰੂ` after a cache update, a disputed `ਵਾਰ ਗੁਰੂ` split that
must not settle, and actual beam promotion when later acoustic evidence
supports a previously lower-ranked interpretation. Those decoder tests use
controlled hypotheses/logits; they are distinct from the captured live audio,
which did not contain that specific word. Additional tests cover a boundary
word finishing at six seconds, confirmation exactly at the eight-second cap,
an unfinished word continuing after the cap, predictor checkpoints ahead of
the encoder cache, and replay that cannot append a suffix to a settled word.
Cache waiting affects finalization, while partial updates keep their cadence.


## Full human-reference benchmark

The supplied Hugging Face recording, Bhai Agaykar Singh, 09-02-2025,
was downloaded together with `aligned_transcript.json`. The entire
3,231.6865625-second recording (53 min 51.7 s) was replayed continuously
in causal 800 ms packets. All 273 verified sections and 1,521 words were
scored, including the rapid recitation passages. The baseline was commit
`eae1d27de167926cfc34d060734b94b8c69aa557`; both runs used the streaming
model and four-thread CPU inference. The prior was selected on an initial
five-minute pilot and then evaluated over the complete recording. No
reference words or timestamps were passed to the decoder.

| Entire recording | Baseline | Updated decoder |
| --- | ---: | ---: |
| Word error rate, all output | 19.86% | 16.24% |
| Substitutions | 244 | 184 |
| Deletions | 44 | 46 |
| Insertions | 14 | 17 |
| Character error rate, including spaces | 7.98% | 7.17% |
| Supplementary verified-window WER | 19.40% | 15.58% |
| Median processing per update | 468 ms | 469 ms |
| p95 processing per update | 665 ms | 696 ms |
| Total CPU replay time | 1,889.314 s | 1,901.375 s |
| Real-time factor (lower is faster) | 0.585 | 0.588 |
| Samples / encoder frames retained | 51,706,985 / 80,793 | 51,706,985 / 80,793 |
| Peak pending audio | 8.0 s | 8.0 s |

The primary word error rate improved by 18.2% relative, chiefly through
fewer spelling substitutions. This does **not** establish human-level
accuracy or elimination of missing words: deletion count rose by two,
and rapid recitations remain difficult for the model. The final eight-word
`ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਹਿ` section was exact in both runs.
All 807 cache events were checked for four-second increments, the
four-to-eight-second pending window, and one second of audio after each
confirmed word-end marker. The full replay is causal and unpaced, rather
than a 53-minute wall-clock WebSocket session.

The full-replay timing includes the previous display calculation. The
subsequent display optimization has exact text/word-timestamp equivalence
coverage and is exercised by paced integration tests. For a synthetic
1,350-word transcript, median snapshot preparation fell from 12.44 ms to
0.071 ms; this is a display microbenchmark, not model inference timing.

Complete transcripts, exact error accounting, audio/reference SHA-256
hashes, source URLs, and runtime provenance are checked in:

- [Baseline report](validation/verified-kirtan/baseline.json)
- [Updated report](validation/verified-kirtan/candidate.json)
- [Section comparison CSV](validation/verified-kirtan/sections.csv)
- [Updated complete transcript](validation/verified-kirtan/candidate.txt)

That revision’s 31 regression tests passed. The original 52.0126875-second MP3
was also replayed over a paced WebSocket using 100 ms packets. HTTP and
WebSocket final text, token IDs, sample counts, frame counts and cache
decisions matched. All 65 partial updates retained complete settled words;
median delivery lag was 417 ms, p95 652 ms, maximum 774 ms. Full-file
inference took 21.556 seconds. The final word was retained after EOF.

A fresh live.sgpc.net capture was tested with the updated decoder: all
108.1585625 seconds, 1,730,537 samples and 2,704 encoder frames were
retained. HTTP and paced 100 ms WebSocket packets produced identical final
text, tokens and cache decisions, with 135 partial updates and 26 deferred
cache steps. Median delivery lag was 458 ms, p95 627 ms, maximum 783 ms;
the peak pending window was eight seconds. Full-file inference took
54.056 seconds. This validates live delivery and parity; the capture has
no human reference. [Attached paced results](validation/verified-kirtan/attached-paced.json)
and [fresh live paced results](validation/verified-kirtan/live-paced.json)
include hashes and complete final transcripts.

The complete app was restarted with the saved Linux FFmpeg/one-worker
instructions. Its live source and ASR connected, all five web/health/state
endpoints returned successfully, and five live caption messages showed the
partial progression `ਚਰਨ` → `ਚਰਨਾਰਬਿੰਦ` → `ਚਰਨਾਰਬਿੰਦ ਭਜਨੰ`. An initial
check expecting 20 text changes timed out during an unchanged hypothesis;
ASR audio updates continued, and the subsequent changing-caption check
passed. Startup source buffering initially created a backlog, which caught
up to the live edge. [Complete app check](validation/verified-kirtan/main-app-check.json)
records the current-instance endpoints and caption evidence.


## First 20-minute comparison with the offline branch

The updated 24-beam decoder was validated on the same 1,200-second recording,
99 human sections and 572 words used for the pushed offline-model comparison.
With only word-final onkar and sihari ignored, streaming WER improves from
13.29% to 10.84%, with deletions falling from 25 to 8. Offline scores 11.01%
WER and 14 deletions, but has better CER: 7.37% versus streaming 8.18%, excluding
spaces. Streaming insertions increase from 12 to 18; the narrow WER win is one
word error. Full vowel-preserving scores, outside-tuning results and exact
recoveries are retained in the report.

All 38 regression tests pass. The final entire 20-minute replay retains all
samples and frames and exactly matches the earlier 24-beam text, tokens and
cache decisions. Pure computation reuse reduces its maximum update time from
2.739 to 1.569 seconds. Actual paced API parity passes on both the supplied MP3
and a 120-second fast passage. Fast-passage median lag is 424 ms; p95 is 2.088
seconds and peak is 3.226 seconds, down from 11.710 seconds before the latency
optimizations. Final lag recovers to 0.934 seconds. Brief delays remain during
heavy recitation despite the unchanged 800 ms audio refresh.

See [the complete report](validation/stream-word-retention20m/README.md),
[highlighted transcript differences](validation/stream-word-retention20m/diff.html),
and [all section comparisons](validation/stream-word-retention20m/sections.csv).
The source, runtime, audio and reference hashes, rejected experiments, complete
transcripts, every paced partial and current live readiness evidence are included.

## KenLM models and complete-word commitment

The supplied 3-, 4- and 5-gram KenLM binaries are available through Git LFS.
The selected model and tuning are documented in [KENLM_ASR.md](KENLM_ASR.md),
with the current [section-24 accuracy and latency comparison](validation/kenlm-section24/README.md)
and [earlier 20-minute measurements](validation/kenlm20m/README.md).

KenLM keeps the complete model-order word history across cache promotions.
The decoder also keeps a word partial when one viable spelling continues a
shorter token prefix, such as `ਗੁਰ` becoming `ਗੁਰੂ`. Shared letters alone can
no longer be committed while a retained beam continues that word. This applies
to both adaptive word confirmation and the hard eight-second audio limit;
the encoder can advance by four seconds while the entire ambiguous word stays
in its acoustic beam. The partial refresh remains 800 ms.
