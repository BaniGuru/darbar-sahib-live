# Bani progression and shabad interruption validation

The new matcher loads ordered routes for Asa Ki Vaar, both Aarti versions and both Rehras versions. It joins `banis` → `bani_lines` → `panktis` → `lines`, ordering by `line_group` then `lines.order_id`. The old normal-shabad scoring functions are unchanged; fuzzy word comparisons have a bounded cache. Outside-shabad switching keeps the original high-confidence/two-hit rule.

## Real recording replays

All recordings were decoded to mono 16 kHz audio and transcribed causally with the existing 24-beam, weight-0.3 Gurbani 3-gram ASR. Its model and streaming code are unchanged. The old and new matchers received exactly the same recorded ASR messages. Every production-cadence 800 ms update and final EOF flush was replayed; no undecoded terminal partial was counted. All decoded samples and encoder frames were retained.

| Recording | Audio tested | Samples | Updates including EOF | Auto-next events | Heard bani pankti IDs |
|---|---:|---:|---:|---:|---:|
| Supplied Asa Ki Vaar | 600 seconds | 9,600,000 | 751 | 13 | 26 |
| Supplied Aarti | Full 729.1711875 seconds | 11,666,739 | 912 | 40 | 51 |
| Actual-audio inserted shabad | 163 seconds | 2,608,000 | 204 | 4 | 7 |

The old matcher has no automatic ordered-bani steps. The new steps display the following pankti before needing two recognized words from it. Counts include repeats and are behavior evidence, not accuracy percentages. Aarti selected both the short and longer routes as compatible/exclusive lines were heard; it does not impose one variant on the entire performance.

New matcher p95 update costs were 180 ms (Asa), 152 ms (Aarti) and 194 ms (splice). Total matcher replay computation was 43.28, 20.81 and 13.60 seconds respectively; legacy scoring with the same comparison cache took 39.58, 24.30 and 13.60 seconds. These measurements include replay overhead and were run with concurrent validation workloads. They are not controlled end-to-end live latency measurements or whole-pipeline real-time guarantees. The ASR remains the dominant work on these recordings.

The [source/input hashes and summary](summary.json) include exact audio SHA256, sample/frame coverage and matcher/ASR source hashes. They identify the old scoring-function bodies verified unchanged against the comparison base commit.

## Actual-audio pauri interruption

The splice uses the supplied Asa recording from 210–286.4 seconds, inserts the previous human-verified Bhai Agaykar Singh recording from 1074–1112 seconds, then resumes Asa from 286.4–335 seconds. This produces 163 seconds of real audio; no reference text was supplied to recognition or the matcher.

The first pauri's final pankti `N5Y1` (“ਕਰਿ ਆਸਣੁ ਡਿਠੋ ਚਾਉ”) was recognized with a consonant error (“ਜਾਉ”). Strong remaining terminal context still confirmed its ending. At splice time **72.8 seconds**, the display predicted the next Asa pankti `5N5B`, before inserted audio began at **76.4 seconds**.

At **82.4 seconds**, the existing matcher switched to shabad **K3Q** (“ਜੋਗੁ ਬਨਿਆ ਤੇਰਾ ਕੀਰਤਨੁ ਗਾਈ”), in normal shabad mode. This is **6 seconds after insertion**. The source returned to Asa at **114.4 seconds**, and bani following resumed on `5N5B` at **134.4 seconds**, **20 seconds after return**. Reacquisition depends on noisy ASR evidence and the preserved shabad-switch confirmation rule; it is not immediate.

[Fixture source ranges/hash](insertion-fixture.json) and [passing transition assertions](insertion-audit.json) retain these checks.

## Tests and live readiness

All **69 regression tests pass**, including 18 new matcher tests. They cover:

- Exact group/order loading and progression through both complete Rehras routes, both Aarti routes and Asa's route.
- End-triggered next-line display and first-word confirmation; ending plus following words in one update.
- Short/long variant evidence, shared lines, end-of-version behavior and repeated IDs in different groups.
- Sung repeats, damaged terminal words, `ਰਹਾਉ`, generic invocations and unchanged/revised ASR updates.
- Ordinary outside-shabad switching and resumption, including a shabad that also belongs to another bani.

[Full test output](tests.txt). The supplied audio tests exercise Asa and Aarti; Rehras has database-route tests, with no Rehras recording supplied. Current full-app readiness is captured separately in `app-readiness.json`.

## What these results establish

These are real-audio recognition and progression/switching tests, not a verified pankti-accuracy or WER measurement. No human-aligned transcript was supplied for the SikhNet recordings. ASR errors can still delay matching or select incorrect lines; held/displayed coverage does not establish correctness. Predicted lines are not counted as heard lines. The whole-recording comparison preserves normal-search fallback rather than forcing every recognized word into a bani.

## Review and reproduce

- [Readable transition comparison](comparison.html): ASR tails, canonical database speech, ordered-bani predictions and legacy results.
- [Asa replay](asa.json), [Aarti replay](aarti.json), [splice replay](insertion.json): complete ASR outputs, every matcher state, transitions, source hashes and timings.
- Plain transcripts: [Asa](asa-transcript.txt), [Aarti](aarti-transcript.txt), [splice](insertion-transcript.txt).
- [Implementation and commands](../../BANI_MATCHING.md).

Audio download URLs are in [summary.json](summary.json). Recordings/PCM and the raw gzip ASR captures remain outside Git in `/workspace/bani-matching-validation`. Reproduce with `scripts/capture_stream_asr.py` and `scripts/replay_bani_matching.py`; matcher replay never receives an expected bani or a reference transcript. The fixture ranges make the splice reproducible from the original sources.
