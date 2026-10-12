# Earlier 20-minute KenLM evaluation

These complete 1,200-second causal replays were measured **before the subsequent whole-word caption-boundary fix**. They are historical evidence for scoring selection, not measurements of the final decoder. The current [section-24 comparison](../kenlm-section24/README.md) reruns all three model orders after that fix.

The same 19,200,000 mono 16 kHz samples and 99 complete human-verified sections were used for every row. The reference contains 572 words. WER ignores only terminal onkar ੁ and sihari ਿ; CER also excludes spaces. Recognition receives no reference text or timestamps. Whole-output scoring counts words in unannotated gaps.

| Decoder | WER | CER | Substitutions | Deletions | Insertions | RTF |
|---|---:|---:|---:|---:|---:|---:|
| Previous streaming | 10.84% | 8.18% | 36 | 8 | 18 | 0.546 |
| Previous offline | 11.01% | 7.37% | 37 | 14 | 12 | See baseline report |
| 3-gram, observed trigram evidence, weight 0.3, bonus 0 | **10.14%** | **7.47%** | 29 | 10 | 19 | 0.559 |
| Same, bonus 0.25 | 10.14% | 7.51% | 29 | 10 | 19 | 0.570 |
| Probability scoring, weight 0.15, bonus 10 | 16.96% | 13.27% | 36 | 2 | 59 | 0.572 |

Zero-bonus observed-trigram evidence was selected. It recovered four earlier omissions (`ਸਚਾ ਢੋਆ ਆਪਣੇ ਚਰਣ`) but did not eliminate all missing words: ten deletions remain versus eight in the previous streaming baseline. Eight new deletions had previously been substitutions. The total word error improved, but fewer deletions cannot be claimed. Stronger probability fusion was rejected because its 59 insertions outweighed recovered words.

Pilot ranges were 0–120 and 680–800 seconds. On sections outside those tuning ranges, WER was 8.56% versus 8.29% before KenLM. This single recording and the supplied LM's unknown training provenance do not establish independent held-out improvement.

## Speed and boundary evidence

Selected processing took 670.82 seconds for 1,200 seconds of audio, 2.33% above the previous streaming run. Median update cost was 432 ms, p95 683 ms and maximum 2.315 seconds. This provides average throughput headroom but does not guarantee low latency on every passage.

The earlier paced supplied 52-second recording passed token/text/cache parity, with p95 lag 0.579 seconds and maximum 0.652 seconds. The earlier paced rapid-recitation passage (680–800 seconds) also passed parity, but p95 delivery lag was **6.73 seconds** and maximum **8.49 seconds**, versus 3.23 seconds in the previous branch's test. These are pre-boundary-fix measurements. The latest section-24 paced check is reported separately and does not cover this heavier passage.

## Artifacts

- [Highlighted previous-stream/offline/KenLM diff](diff.html), [99 section rows](sections.csv), [summary and recovered/missing words](summary.json).
- [Selected earlier replay](trigram.json) and [transcript](trigram.txt).
- [Rejected probability replay](probability.json), [rejected bonus replay](bonus025.json), [pilot profiles](tuning.json).
- Previous streaming/offline measurements are preserved in [the earlier baseline report](../stream-word-retention20m/README.md).
- Paced evidence is stored as `provided-paced.json` and `rapid-paced.json` alongside these reports.

Every replay JSON includes its actual model and source hashes. The legacy `trigram` scoring mode means observed matches of order three or higher; for this 3-gram binary it is equivalent to the current `ngram` setting. The summary's `bonus025` row is the rejected extra-word-bonus profile.
