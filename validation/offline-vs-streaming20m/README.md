# Offline versus streaming: first 20 minutes

Identical 1,200-second input, 19,200,000 samples, 99 human-verified sections and 572 reference words. Both models use CPU inference. No reference text or timestamps enter decoding or spelling restoration.

| Path | WER | Substitutions | Missing words | Extra words |
|---|---:|---:|---:|---:|
| Original main greedy, same overlap handling | 39.51% | 183 | 40 | 3 |
| Updated offline, raw ASR | 35.14% | 175 | 14 | 12 |
| Updated offline, restored spelling | 16.78% | 70 | 14 | 12 |
| Current streaming | 18.36% | 68 | 25 | 12 |
| Streaming with matched spelling restoration (analysis only) | 17.48% | 63 | 25 | 12 |

The improved offline pipeline wins on this sample for displayed-word accuracy and word retention. It has 11 fewer deletions than streaming. With identical spelling restoration on both, offline has four fewer total word errors (96 versus 100). The raw offline decoder remains less accurate than the streaming decoder: much of its WER improvement comes from restoring final short vowel marks using independent corpus context. This is a pipeline comparison, not evidence that the offline acoustic model alone is superior.

Offline inference sees up to 30 seconds per uncached window, with five seconds of context on both sides of each 20-second core. It can use future audio. Streaming replays every 800 ms and caps pending audio at eight seconds. These full-recording results do not establish equal live latency or partial accuracy. Benchmark jobs ran concurrently; elapsed times are not a controlled speed comparison.

The original main endpoint attempts unbounded full-recording attention. Its baseline here uses the bounded adapter to make evaluation feasible. Both greedy and updated offline runs use the same model weights, window schedule and anchor reconciliation; the initial ownership-only adapter scored 39.86% WER with 44 deletions. Updated anchor handling plus greedy scored 39.51% with 40 deletions.

The spelling restorer changes only final ੁ/ਿ when an exact three-word stem context is supported by the existing Gurbani database with at least 80% vote agreement. It cannot add/remove words, change consonants, change pronounced vowel marks, or repair false word splits. All 123 corrections are recorded; raw text, raw tokens, and original word timestamps are retained. Word count and timestamps were checked unchanged by restoration.

The first five minutes were used for pilots. Across reference sections beginning at or after 300 seconds (469 words), offline scored 16.20%, streaming 20.68%, and the matched streaming spelling control 19.62%. Repeated passages can make section attribution ambiguous; primary scores use exact global word alignment.

Main base: `e81db754638ca79706ec546b171795494ed9e53c`. Streaming control: `db2632a93f8e1aabad23890286b14542278cbd78`. Offline uses main’s model/ ONNX assets; the streaming control uses that pushed streaming revision and its own cached-model assets.

## Transcript diff

[Side-by-side word diff](diff.html) · [All 99 sections as CSV](sections.csv) · [Offline transcript](candidate20m.txt) · [Raw offline transcript](offline-raw20m.txt) · [Streaming transcript](streaming20m.txt)

The HTML highlights word insertions, missing words, and substitutions against the human reference. It also includes the direct offline-versus-streaming diff. JSON reports preserve exact errors, audio/reference hashes, word timestamps and recognition metadata.
