The player displays one pankti at a time, starting at its first recognized
word. The earlier two-pankti presentation is removed, including its second
DOM entry and pairing styles. Frontend cache version is **18**.

Matching still uses words received ahead of delayed playback to confirm a
line. Once confirmed, its boundary is attached to the first supporting
ASR word, rather than the later word that established confidence. Every
word in a fast ASR batch is considered, preserving intermediate lines.
Ambiguous phrases retain the existing matching safeguards. This is an ASR
word timestamp, not a guarantee of exact phonetic onset.

The shared recognition handler prepares a confirmed playback timeline and
adds `pankti_id`, `shabad_id`, bani metadata and `audio_timestamp` to its
timed words. Internal `bani-auto-next` predictions are excluded from this
playback timeline: the listener keeps the heard line until the next line's
confirmed first-word boundary. Recognition ahead of playback can confirm
that boundary before it is heard, without showing an unheard prediction.

`/api/live/state` returns:

- Scalar canonical text and translations for the listener's current line.
- `display_panktis`, a compatibility array containing zero or one line.
- `playback_words`, recent heard words with prepared pankti IDs.
- `pankti_schedule`, the current event and confirmed upcoming events within
  the configured playback delay (at most 64 events), each containing
  `audio_timestamp` and `line`. Waiting-after-greeting events are included.

The schedule participates in the state revision, so a newly confirmed
future boundary is delivered even when the current pankti is unchanged.
Each listener's request reads the shared timeline; it does no matching.
The browser renders scalar text, ignoring any stale paired array.

Between the one-second state polls, the browser schedules the next boundary
against `audio_start_timestamp + audio.currentTime - playbackBase`. Timer
callbacks recheck the actual media clock, and media time/seek/rate events
resynchronize it. Buffering or pausing cannot advance to a future pankti
just because wall time passes. The same scheduling makes the greeting's
waiting artwork follow the audio and keeps it until a heard shabad returns.

The acoustic model, KenLM configuration, four-second recognition blocks
and ten-second playback delay remain unchanged. If recognition arrives
after its playback boundary, the newly available confirmed line appears
on receipt; the browser cannot prepare evidence it has not received.

Verification:

- **133 Python tests** pass, including single-line first-word transitions,
  first-word pankti IDs before the second word is heard, internal bani
  predictions, future schedule revisions and shared prepared-poll reuse.
- Chromium at **1280×800, 390×844 and 320×700** checks one-line rendering,
  ignoring stale paired arrays, between-poll transitions while subsequent
  polls return 204, audio stalls, timed greetings and existing loading,
  recovery, calendar and listener behavior. Actual SGPC AAC playback,
  manual stop and listener counting pass separately.
- Causal ASR fixture replays cover the requested recording at
  **02:35:00–02:40:00** and **12:08:36–12:11:00**, including the original
  16-second pre-roll and 12-second post-roll. All **301 + 145** one-second
  listener samples contain at most one displayed pankti, identical to the
  scalar current ID, with no future scalar timestamps.
- Exact unambiguous ASR word-pair consistency remains **202/202** for Asa
  Ki Vaar and **46/47** for the fast passage (old baseline `f48d81a`:
  179/202 and 32/47). Three cyclic refrain pairs remain excluded; the
  remaining end-of-range cross-boundary discrepancy is retained in the
  audit. These scores measure following ASR words, **not WER/CER or human
  transcription accuracy**. The model was not retranscribed or retuned.

Evidence is in [validation/single-pankti](validation/single-pankti).
The existing compressed ASR captures and recording provenance remain in
[validation/rapid-pankti](validation/rapid-pankti). Reproduce the causal
replays with the commands in [RAPID_PANKTI_FOLLOWING.md](RAPID_PANKTI_FOLLOWING.md);
the validator now audits the confirmed playback timeline. Historical
pairing helpers remain only to support earlier baseline replays.

Pull `no-audio-fix` and restart the production service to load the new
Python timeline preparation and frontend version. No dependency update is
required. Cloud validation does not deploy the external production server.
