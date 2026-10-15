# Landing screen between shabads

The player shows the original `public/darbar-sahib-live.png` landing layout,
with listeners, SGPC date and India time. It does not show a raagi name or use
the proposed raagi artwork. The duties JSON updater remains independent.

During playback, the landing layout appears when:

- No supported shabad words have arrived for **60 seconds**.
- The closing pauri `ਵਡੀ ਹੂ ਵਡਾ ਅਪਾਰੁ ਤੇਰਾ ਮਰਤਬਾ` is recognized.
- The closing salok `ਪਵਣੁ ਗੁਰੂ ਪਾਣੀ ਪਿਤਾ ਮਾਤਾ ਧਰਤਿ ਮਹਤੁ` is recognized.
- The Khalsa/Fateh greeting or repeated `ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ` is recognized.
- Ardas is recognized from bani membership or a strong Ardas phrase.

Minor ASR spelling variations are accepted. The two closing compositions are
identified by their opening phrases in the database, and all their subsequent
panktis keep the landing screen visible. The current database supplies two
closing shabads (14 indexed panktis) and 27 Ardas panktis. Common two-word
fragments alone do not classify Ardas or a closing pauri.

A new confirmed shabad outside these passages restores the single-pankti view
at its first supported word. While following Ardas or a closing passage,
ambiguous names/repeated words do not resume an unrelated verse: a strong new
phrase must support four distinct words. Repeated ਹਰਿ therefore cannot count
as four separate pieces of evidence. The closing greeting releases this extra
context check. Ordinary shabad matching keeps its existing rules.

All transitions follow the listener's **audio clock**, including between
one-second state polls. Recognition is already ahead of the ten-second-delayed
playback. Future recognition cannot dismiss waiting early, and stalled media
cannot advance the display. Idle messages and retained transcript text do not
renew an old pankti. The generic timeout is measured from the latest supported
word end, not from server processing time or the last poll.

The API exposes `waiting_for_shabad`, `waiting_reason` and prepared
`pankti_schedule` boundaries. There are no raagi fields or duties-file reads in
the player path. Initial welcome, pending-play spinner/Starting soon, clock,
listeners and connection recovery keep their existing behavior. UI cache
version is **20**.

## Recording verification

The source is the requested October 14, 2026 recording, SHA256
`23d8c9f385f9ecbea94c52d8765895b04cc859cb1d30f43092dd36553851f729`.
Offsets below are file playback times. The filename states 02:00 India as the
recording start; adding file offsets maps the historical clock.

Production four-second block ASR with the existing KenLM 3-gram/24 beams ran:

- 03:41:19–03:43:47, including the requested Ardas start at 03:41:35.
- 06:59:18–07:02:14, covering the complete 06:59:34–07:02:02 request.

Captures include 16 seconds of preceding context and 12 seconds after each
listener interval. They are replayed causally, ten seconds behind received
audio. The Ardas replay preserves the final Anand Sahib line, then stays on
the landing screen from the Waheguru transition at approximately 03:41:45.
The original implementation briefly selected unrelated verses from prayer
names; the updated matcher removes those flashes. The second requested
interval stays on the landing screen through both closing passages, the
Fateh greeting and the following quiet/ambient audio, with no raagi name.

These are real acoustic captures and presentation checks, not human WER/CER.
No acoustic-model, beam, KenLM, audio-buffer or CPU-setting change was made.
See [validation/waiting-screen](validation/waiting-screen) for reports and
compressed causal messages. Source audio/PCM are kept outside Git.

Reproduce presentation without another acoustic run:

```bash
.venv/bin/python scripts/replay_waiting_screen.py --events validation/waiting-screen/ardas.jsonl.gz --capture-report validation/waiting-screen/ardas-capture.json --recording-start 2026-10-14T02:00:00+05:30 --start 13295 --end 13415 --expect-waiting-from 13305 --report /tmp/ardas-presentation.json
.venv/bin/python scripts/replay_waiting_screen.py --events validation/waiting-screen/role-ending.jsonl.gz --capture-report validation/waiting-screen/role-ending-capture.json --recording-start 2026-10-14T02:00:00+05:30 --start 25174 --end 25322 --expect-waiting-from 25174 --report /tmp/role-presentation.json
```

Pull `no-audio-fix` and restart the existing application service for the new
server logic. No new package or artwork is required. Cloud validation/push
does not deploy to the production server or activate its duties cron job.
