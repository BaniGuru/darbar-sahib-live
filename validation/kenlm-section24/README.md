# KenLM comparison through section 24

Selected **3-gram, weight 0.3, `ngram` mode, zero word bonus, 24 beams**. All three models produced identical raw transcripts; the smaller 3-gram wins the accuracy tie.

## Input and scoring

Every current row was rerun **after the whole-word caption-boundary fix**, on the same public Bhai Agaykar Singh recording from **0 to 340.8 seconds**, through `section-24` (327.72–340.8 seconds). Input is exactly **5,452,800 mono 16 kHz samples**, 24 human-verified sections and **118 reference words**. Each causal replay processes 426 updates at 800 ms. The recognizer receives audio only; reference text and timestamps are scoring-only.

Primary scores ignore only word-final **onkar ੁ and sihari ਿ**. Internal marks, long vowels, repetitions and word boundaries remain significant. CER excludes spaces; JSON also records CER including spaces and vowel-preserving scores. All output counts, including words in unannotated gaps; an insertion need not be a hallucination.

## Controlled comparison

All rows use the same acoustic model weights, corpus prior weight 0.6, 24 beams and predictor caching. Only KenLM order changes between the three main rows.

| Model | WER | CER | Substitutions | Missing words | Extra words | Processing | RTF | p95 update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6.78% | 4.01% | 4 | 0 | 4 | 180.70 s | 0.530 | 0.579 s |
| 3gram | 5.93% | 3.79% | 3 | 0 | 4 | 186.50 s | 0.547 | 0.614 s |
| 4gram | 5.93% | 3.79% | 3 | 0 | 4 | 182.18 s | 0.535 | 0.578 s |
| 5gram | 5.93% | 3.79% | 3 | 0 | 4 | 181.86 s | 0.534 | 0.581 s |

RTF is processing time divided by 340.8 audio seconds; below one means faster than playback. Every candidate also has p95 update cost below the 800 ms refresh. Timings vary, so small differences do not establish a universal speed ranking. Files are 28.0, 45.2 and 60.6 MB respectively. Full native model-order history is retained: two, three or four completed context words, including across cache promotions.

## Whole-word cache correction

The paced test exposed a real bug in the earlier decoder: it could commit `ਗੁਰ` while retaining a compatible `ਗੁਰੂ` hypothesis whose remaining vowel continued the already committed word. The decoder now keeps that whole word partial when a viable longer spelling continues its token prefix. Both adaptive confirmation and hard-limit promotion apply this check. Encoder caches can still advance four seconds while the ambiguous word stays in its acoustic beam; pending audio remains capped at eight seconds. The fix does not force a particular spelling or change acoustic weights.

## Tuning and limits

Earlier, pre-fix 0.3-weight runs tied at 5.08% WER / 3.56% CER. Increasing the weight to 0.6 worsened all three orders to 6.78% WER / 4.90% CER. Maximum-order-only scoring for 4/5-gram tied the earlier conservative result. Those complete experiments are preserved in [pre-boundary-fix](pre-boundary-fix/README.md) and excluded from current model selection. Current scores above include the commitment correction; preserving stable captions also changed one spelling error on this recording.

This 5-minute-41-second prefix does not establish 4/5-gram performance over the whole recording. [Earlier 20-minute 3-gram testing](../kenlm20m/README.md) records useful settings and regressions, including up to 8.49 seconds of lag on a fast passage before this boundary fix. The supplied models' training corpus was not provided; this is a recording-specific comparison, not an independent held-out accuracy result.

## Artifacts and reproduction

- [Highlighted transcript diff](diff.html), [all 24 sections](sections.csv), [selection and timings](summary.json).
- [3-gram](3gram.json), [4-gram](4gram.json), [5-gram](5gram.json), [no-LM control](control.json): full transcripts, tokens, word times, metrics, cache events and source/model hashes.
- [Provenance](provenance.json), [51 regression tests](tests.txt).

```bash
ASR_KENLM_PATH=kirtan_stream_model/gurbani_5gram.bin \
ASR_KENLM_MODE=ngram ASR_KENLM_WEIGHT=0.3 ASR_KENLM_WORD_BONUS=0 \
.venv/bin/python scripts/evaluate_reference_asr.py \
  --audio /workspace/kenlm-validation/section24.wav \
  --reference /workspace/kenlm-validation/reference24.json \
  --ignore-final-short-vowels --report /tmp/5gram24.json
```

Replace the filename for 3/4-gram. `ngram` uses observed matches of order three or higher, including native backoff; `full_order` requires the maximum model order. Set `ASR_KENLM_WEIGHT=0` for the control. Native log10 probabilities are converted to nats; conservative evidence is a capped conditional/unigram ratio, not plain probability fusion.

## Actual paced streaming

The selected 3-gram replayed all 340.8 seconds at playback speed with 100 ms WebSocket packets. All **426 partial updates and 84 cache commits** passed exact text/token/cache-event parity against the completed causal replay. Every sample and encoder frame was retained, committed captions remained immutable whole-word prefixes of the final transcript, and the peak pending window was **8 seconds**. Median delivery lag was **428 ms**, p95 **594 ms**, maximum **779 ms**. First text appeared at audio time 1.6 seconds (wall time 1.895 seconds). These are measured network-delivery timings, separate from processing-cost simulation.

[All paced messages and final output](paced.json) retain timing, text and word records. Reproduce with the selected standalone server on port 8001:

```bash
.venv/bin/python scripts/validate_streaming_asr.py /workspace/kenlm-validation/section24.wav \
  --url http://127.0.0.1:8001 --packet-ms 100 \
  --full-result /workspace/kenlm-validation/order324.json --report /tmp/paced24.json
```

The complete live application also passed all five page/health/state endpoints, source/ASR connectivity, selected KenLM health settings and five changing caption messages. [Live readiness evidence](app-readiness.json) does not measure human-reference live accuracy; source buffering can create startup lag.
