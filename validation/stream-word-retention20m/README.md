# Streaming word retention: first 20 minutes

The cached RNNT decoder now retains up to 24 competing hypotheses. Its predictor and acoustic projections are factored from the unchanged ONNX graph and cached only for identical tensor inputs. Corpus scores and predictor caches are bounded, and sibling hypotheses share immutable predictor states. The wider search recovers weak, rapidly recited words while retaining the 800 ms refresh, four-second encoder cache steps, adaptive word-end lookahead, and eight-second pending-audio cap.

## Exact same recording and scoring

All three results cover the first **1,200.0 seconds**, **19,200,000 PCM samples**, **99 complete human-reference sections**, and **572 reference words** of Bhai Agaykar Singh's 09-02-2025 recording. The public source is the `speech-data-darbar-sahib/Bhai Agaykar Singh (3.20pm to 4.20pm) 09-02-2025` directory in [sukhpalsingh/speech_data_recordings](https://huggingface.co/datasets/sukhpalsingh/speech_data_recordings). Reference sections ending after 1,200 seconds were excluded; the final included section ends at 1,198.63169 seconds. The WAV and reference SHA-256 values are in each JSON report.

Requested normalization removes **only word-final onkar (`ੁ`, U+0A41) and sihari (`ਿ`, U+0A3F)**, after NFC, punctuation and standalone verse-number removal. Internal vowels, long vowels, word boundaries and repetitions remain significant. `ਵਾਰ ਗੁਰੂ` cannot match `ਵਾਹਿਗੁਰੂ`. CER below uses Unicode code points and **excludes spaces**; the reports also retain CER including spaces and scores preserving all vowel marks.

| Model | WER | CER, excluding spaces | Substitutions | Deletions | Insertions |
|---|---:|---:|---:|---:|---:|
| Previous streaming, `db2632a`, four beams | 13.29% | 9.46% | 39 | 25 | 12 |
| Pushed offline-model, `1c3dab5` | 11.01% | **7.37%** | 37 | 14 | 12 |
| Updated streaming, 24 beams | **10.84%** | 8.18% | 36 | **8** | 18 |

The updated streaming decoder has 14 fewer total word errors than its predecessor, an 18.42% relative WER reduction, and 68% fewer deletions. It narrowly beats offline WER by **one word error**, and has six fewer deletions. **Offline still has better character accuracy.** Streaming insertions increased by six. This is a measured improvement on this recording, not general superiority over offline.

Eleven previously deleted reference words now match exactly. Eight other old deletions became substitutions rather than correct words, six remain missing, and two new deletions appeared. Exact recoveries include `ਕਾ ਨਾਮ ਧਿਆਇ ਕੈ ਹੋਹ ਹਰਿਆ`, `ਪੰਜੇ ਬਧੇ ਮਹਾਬਲੀ ਕਰ`, and `ਤੇ`. The newly missing words are `ਫਲ` and `ਚਰਣ` in the rapid passages. See [summary.json](summary.json) for their section times and complete accounting.

With vowel marks preserved, WER is 18.36% for previous streaming, 16.78% for the restored offline display, and 15.21% for updated streaming. Offline's raw and restored outputs score identically under the requested final-mark normalization; the spelling restorer does not influence the normalized comparison.

## Selection and limitations

Search widths 4, 8, 16 and 24 were tested on the 680–800-second fast-recitation excerpt. Their causal cold-start pilot WERs were 28.49%, 27.33%, 22.09% and 18.02%, respectively. Wider search is enabled globally, with probability pruning. Silence-score bias, stronger corpus weighting, shorter normalization history, longer compulsory lookahead and adaptive narrower beams were tested and rejected because they worsened pilot accuracy or lost recovered words. No reference text or timestamps enter inference, and no benchmark words are hardcoded.

The production candidate was then replayed through **all 1,500 actual 800 ms updates**, with no future samples available to each update. It returned all **30,000 encoder frames**, flushed the terminal word, and stayed within the eight-second cap. It took 655.579 seconds, or 0.546 times real time; median update processing was 432 ms, p95 638 ms, and maximum 1.569 seconds. A maximum denotes an occasional slow update, not the typical delivery delay.

The 21 tuning sections contain 172 words; the other 78 sections contain 400. On the tuning sections, full-recording WER changes from 26.74% to 16.28%. Outside them, WER changes from 7.50% to 8.50%, driven by six extra insertions despite two fewer substitutions. This limitation is retained in the report. The full 20 minutes were checked after selecting the beam width; this is not an entirely held-out benchmark.

Primary WER/CER score **every output word**. The reference has unannotated gaps, including the spoken `ਬੋਲੋ ਜੀ` introduction, so an insertion is not automatically a hallucination. Supplementary timestamp-covered scores are in the JSON; they do not replace the primary comparison. Offline uses 20-second cores with five seconds of past and future context on each side, whereas streaming has at most eight seconds of pending audio. Baseline full-run timing was CPU-contended and is not used to claim a speed gain.

## Artifacts and reproduction

- [Highlighted word differences](diff.html): all 99 sections, previous streaming, offline and updated streaming.
- [Section CSV](sections.csv): exact per-section errors and hypotheses.
- [Previous streaming](stream_before.json), [offline](offline.json), [updated streaming](stream_after.json): full text, tokens, word times, both scoring modes, hashes and performance.
- [Summary](summary.json): recovered words, new omissions, tuning and outside-tuning results.

From the repository root:

```bash
.venv/bin/python scripts/evaluate_reference_asr.py \
  --audio /workspace/offline-model-validation/first20m.wav \
  --reference /workspace/offline-model-validation/reference20m.json \
  --ignore-final-short-vowels \
  --report /tmp/stream20m.json
```

`--hypothesis` can rescore saved output. The recognizer receives audio only. `ASR_BEAM_WIDTH=4` selects the old search width; `ASR_CACHE_PREDICTOR=0` uses the original combined graph. Neither changes the model weights.

## Paced API delivery

The original 52.0126875-second supplied MP3 was replayed over HTTP and a paced WebSocket with 100 ms packets, against separate verified baseline and candidate runtimes. Both retained all 832,203 samples and 1,301 encoder frames, produced 65 partial updates, flushed EOF, and returned identical HTTP/WebSocket text, tokens and adaptive cache decisions. The pending window peaked at eight seconds.

| Same-machine supplied clip | Baseline four beams | Updated 24 beams |
|---|---:|---:|
| HTTP processing | 21.414 s | 21.530 s |
| Median paced caption lag | 372 ms | 411 ms |
| p95 paced caption lag | 564 ms | 586 ms |
| Maximum paced caption lag | 709 ms | 664 ms |

The wider beam has comparable delivery speed on this clip. Timing varies by workload; the separate rapid-passage check below exercises heavier decoding. [Baseline paced evidence](control-paced.json) and [candidate paced evidence](candidate-paced.json) contain every partial snapshot and final transcript. Numerical model parity and boundary/EOF/scoring checks passed: **38 tests**, recorded in [tests.txt](tests.txt).


The final 120-second rapid-recitation excerpt also passed paced 100 ms WebSocket/HTTP parity: all 1,920,000 samples, 3,000 frames, 150 partial updates and 29 cache promotions matched. Its peak pending window was eight seconds. Median delivery lag was **424 ms**, p95 **2.088 seconds**, maximum **3.226 seconds**, and final EOF lag **0.934 seconds**. Heavy recitation still creates brief delays, then catches up; this is not a claim that every partial arrives within 800 ms.

The first 24-beam implementation reached 11.710 seconds of peak lag. Enlarging the bounded predictor cache to 16,384 entries, batching cache misses, and reusing complete corpus-prefix scores reduced that peak to 3.226 seconds **without changing any output token or cache decision**. The optimized entire 20-minute replay likewise matches the earlier 24-beam transcript, tokens and cache events exactly. The latter's maximum update processing fell from 2.739 to 1.569 seconds. The corpus scorer preserves its full-text scores exactly, including numeric/unknown fallbacks; 9,267 additional prefix cases were checked.

[Final rapid paced evidence](rapid-paced.json), [initial stress evidence](rapid-paced-before-optimization.json), [pilot selection and rejected configurations](pilot-summary.json) and [runtime/source provenance](provenance.json) retain the measured limits and exact checks.


The full live application also passed readiness checks after restarting the final source: all five web/health/state endpoints responded, live source and ASR connected, health confirmed 24 beams and predictor caching, and five changing captions arrived. [Live readiness evidence](app-readiness.json) verifies delivery and startup; this current live capture has no human reference, so no live accuracy score is claimed.
