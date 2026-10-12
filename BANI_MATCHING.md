# Ordered bani matching

The live matcher follows Asa Ki Vaar, Aarti (both versions), and Rehras Sahib (both database versions) in their bani order. Normal shabad search and its confidence/confirmation rules remain in place.

## Database and progression

The shipped database calls the catalogue `banis`; exports using `banies` are also accepted. Routes join the catalogue to `bani_lines`, then `panktis.id = bani_lines.line_id` and `lines.id = panktis.id`. Ordering is `bani_lines.line_group`, then `lines.order_id`. Occurrences retain both their group and position, including repeated pankti IDs. Existing non-sung heading filtering is retained. A shared invocation alone cannot identify a bani.

Once a bani line is identified, strong ordered terminal words trigger immediate display of its next pankti. The next pankti can therefore appear before any of its words are recognized. A strong first word can confirm the expected next line; it does not need the ordinary two-word fragment match. One ASR update can contain an ending and subsequent words: the ending is anchored separately, and subsequent words remain eligible for matching. Timestamps prevent an old ending from completing a newly sung repeat. Backward repeats need stronger evidence than forward continuation.

`ਰਹਾਉ` and standalone verse numbers are treated as annotations when testing a sung ending. Nearby matching allows omitted invocations and noisy speech. This follows the ASR evidence, not a playback timer; incorrect or missing recognition can delay or misidentify a pankti.

## Aarti and Rehras alternatives

Both routes stay available through shared lines. Evidence is counted from heard lines, never from an auto-next prediction or repeated unchanged updates. The best-supported compatible route supplies the next pankti; exclusive sung lines select their corresponding version, and common lines can rejoin either route. If the selected shorter version ends, the matcher does not invent a continuation from stale longer-version positions. Performances can include material from both Aarti routes.

## Shabad interruptions

When the expected bani words do not match, the existing shabad matcher handles the update. Its outside-shabad switch still requires two high-confidence hits at the existing threshold. Bani predictions do not lock search to the predicted shabad: the active shabad remains the one actually heard. Accepted outside shabads suspend bani following, with context retained for resumption.

A shabad inserted into Asa Ki Vaar can itself appear in Aarti or Rehras. That membership does not start another bani while the recent Asa session is interrupted. The ordinary shabad continues to display; Asa can resume when its words are recognized again. The recent-session hold uses the existing five-minute interval.

## API and implementation

`/api/live/state` retains its existing fields and additionally returns `bani_id`, `bani_name`, `line_group`, `bani_position`, and `match_mode` (`bani`, `bani-auto-next`, or null for ordinary matching). Timeline entries remain anchored to matched source-audio word times. The existing page can display the new pankti without a client-side change.

- `bani_matching.py`: route context, variant evidence, terminal-word confirmation and progression.
- `server.py`: ordered database joins, integration, unchanged shabad fallback and public state/timeline.
- `tests/test_bani_matching.py`: ordering, complete database routes, variants, repeats, invocations, first-word confirmation and interruptions.

Fuzzy token comparisons use a bounded 65,536-entry cache. Their values and the normal shabad scoring algorithm are unchanged. The 3-gram ASR, acoustic model, encoder-cache timing, 800 ms refresh and eight-second pending limit are unchanged by this work.

## Validation and reproduction

See [the recording report](validation/bani-matching/README.md) for the first 600 seconds of the supplied Asa Ki Vaar audio, all of the supplied Aarti audio, and an actual-audio splice that inserts the previous recording after a pauri and then resumes Asa. The report retains the old-matcher comparison, ASR transcripts, source hashes and transition traces. Rehras is tested against both complete database routes; no Rehras recording was supplied.

Capture causal production-cadence ASR updates, with the selected 3-gram defaults:

```bash
.venv/bin/python scripts/capture_stream_asr.py \
  --audio /path/to/asa-ki-vaar.mp3 --seconds 600 --output /tmp/asa10
.venv/bin/python scripts/replay_bani_matching.py \
  --events /tmp/asa10.jsonl.gz --asr-report /tmp/asa10.json \
  --baseline --report /tmp/asa-matching.json
```

Omit `--seconds` for a complete recording. The capture emits every 800 ms partial plus the final EOF flush, retaining all terminal audio. Matcher replays use the same recorded ASR messages for both implementations. They are accelerated causal replays, not paced network latency tests. Predictions are not counted as heard panktis; held/displayed coverage is not an accuracy score. No human-aligned reference was supplied for these new recordings, so their reports do not claim WER or complete pankti accuracy.

Run all regressions with `.venv/bin/python -m unittest discover -s tests`. Public SikhNet fixtures require `media.sikhnet.com` in the environment network allowlist; recordings remain outside Git.
