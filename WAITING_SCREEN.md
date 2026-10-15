# Waiting screen and current raagi

The player returns to the waiting screen after **30 seconds without supported
shabad words**. The deadline follows source audio timestamps and the listener's
delayed audio position, including between state polls. Silence/idle ASR messages
and retained transcript text do not extend an old pankti. A freshly confirmed
shabad resumes the existing single-pankti view at its first supported word.

Repeated `ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ` (including minor spelling variations), the existing
complete Khalsa/Fateh greeting, and recognized Ardas panktis use the existing
waiting artwork immediately at their audio boundary. They do not select the
four-Waheguru verse. Ardas membership comes from `banis` / `bani_lines` joined to
`panktis` and `lines`; the current database supplies 27 Ardas lines. Matching
uses canonical membership or a strong four-word Ardas phrase, rather than
common pairs such as `ਗੁਰ ਨਾਨਕ`. Continued chant/Ardas does not reset the
30-second waiting-screen deadline. Normal shabad matching can resume within
the same ASR message after the special phrase.

After 30 seconds, the API adds the current `raagi`, if available, from
`data/kirtan_duties.json`. This is the structured output of
`scripts/update_kirtan_duties.py`, described in [KIRTAN_DUTIES.md](KIRTAN_DUTIES.md).
`KIRTAN_DUTIES_PATH` optionally overrides the JSON path; relative paths are
resolved against the application directory. The shared lookup reloads on file
replacement/change without restarting the service. Missing, deleted, invalid,
overlapping or incomplete entries return no name, rather than an old name.

Lookup uses `Asia/Kolkata` and the date/time of the audio being heard. Dates
are `DD-MM-YYYY`; times are 24-hour `HH:MM`. Start times are inclusive and end
times exclusive. An empty samapti end applies through that date's midnight;
it never carries yesterday's raagi into a missing date. Genuine schedule gaps
remain gaps. Changes to the duty/name participate in the API revision even
without another ASR update.

## Artwork prerequisite

The new raagi card expects the user's original artwork at
`public/darbar-sahib-raagi.png`. **That file has not yet been supplied as a
downloadable attachment.** An inline preview is visible in the conversation
but is not available to the workspace as file bytes. No replacement artwork
has been generated or substituted.

Until that file is added, `raagi_artwork_url` is null and the player shows the
existing greeting layout, even when the duties contain a name. Missing names
also use that layout. An image download failure falls back to it. The frontend
is prepared to put the raagi in the supplied artwork's centre panel, retain
the date/India clock and right-aligned listeners, shrink long names, and wrap
metadata below the image on phones. Its cache version is **19**.

Adding the exact artwork and visually checking it remains outstanding. Once
present, the server detects the file on its next state request. Browser
regressions automatically use the real file when it exists; otherwise their
explicitly marked aspect-ratio scaffold tests only the UI transitions and
overlay geometry. It is not production artwork or evidence of visual fidelity.

## Validation and deployment

See [validation/waiting-screen](validation/waiting-screen) for the current
test results, browser checks and causal ASR replay comparison. The browser
checks cover initial visits, pending play, recovery, single-pankti display,
media-clock transitions with no new poll body, stalled audio, all waiting
reasons, no duty, image-load failure, listener/date layout, and 1280/390/320px
viewports. No new acoustic transcription or human WER measurement was made;
ASR, KenLM, beam settings and the ten-second audio delay are unchanged.

Pull `no-audio-fix` and restart the existing application service for the new
server logic. Keep the existing cron entry and writable duties JSON location.
No new package is required. This cloud task does not deploy to the production
server, activate cron, or supply missing production duties.
