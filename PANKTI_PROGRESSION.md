The matcher now checks a strong newer phrase even when two shared words
still match the active shabad. Previously, “ਗੁਰ ਰਾਮਦਾਸ” could keep the
player on `99AN` in `SAM`, despite the newer phrase “ਕਾਟੇ ਸੁ ਪਾਪ ਨਰਹੁ ਕੇ ਗੁਰ
ਰਾਮਦਾਸ ਜਿਨ ਪਾਇਓ” identifying `BKS1` in `NJM`. The newer candidate must have
at least three supported words and two distinctive words beyond the old
line, and an outside shabad still requires two consecutive confirmations.
The same evidence can advance a line within the current shabad. Shared
names alone cannot switch the shabad.

Database and ASR search words use the same terminal sihari/onkar/virama
comparison form; displayed canonical Gurbani is unchanged. Global alignment
checks up to three plausible offsets so repeated phrases do not hide the
fresh tail. Playback timestamps use the **spoken Gurbani**, rather than the
Punjabi prose translation, and short active-line matches carry their actual
query start. Ordinary line/backward confirmation and ordered bani following
remain in place.

The complete greeting “ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖ਼ਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫ਼ਤਹਿ” accepts
small spelling differences. It inserts a waiting event at the greeting's
source-audio end, clears matching/bani resume state, and excludes pre-greeting
words from subsequent search. `/api/live/state.waiting_for_shabad` follows
each listener's playback clock. The waiting image/listeners/date persist
until the next matched shabad is heard; future recognition cannot dismiss
it early. A Waheguru refrain or incomplete greeting does not trigger this
transition. Waiting does not pause audio or disable the pause button. Normal
connection recovery retains the displayed Gurbani as before.

Validation: all 117 Python tests pass, including the visible recorded phrase,
shared-name protection, same-shabad evidence, fuzzy greeting, delayed
transition, old-word rejection, waiting lifetime and bani reset. Controlled
Chromium tests pass at 1280×800, 390×844 and 320×700, covering playback
loading/recovery, greeting/next-shabad transitions, actual artwork, India
clock/date and badge layout. Actual SGPC AAC playback, listener counts and
manual stop/reload also pass. See `validation/pankti-progression`.

The supplied 12.8-second screen recording has **no audio track**. Its visible
debug words were replayed against the full 130,840-row database; this is a
matcher reproduction, not a new acoustic WER/CER measurement. A 24-query
matching benchmark on the current cloud CPU had median 0.113 seconds and
maximum 0.146 seconds, including active and global search. These timings
are not a guarantee for the production CPU. ASR model, KenLM, four-second
blocks and ten-second playback delay were not changed.
