The welcome and between-shabad artwork displays the **SGPC modified
Nanakshahi calendar**, with Punjabi month names and numerals, followed by
India time in `hh:mm am/pm`. Both use `Asia/Kolkata`, independently of the
listener's location and the audio delay. The existing local player clock
continues to show the listener's time.

`public/sgpc-calendar.js` contains the published month-start dates for
Nanakshahi 557 and 558. These are taken from the pinned Jantri tables in
[nanakshahi-jantri 1.2.0](https://github.com/janpreet/nanakshahi-jantri):
`Jantri-Nanakshahi-557.pdf` (official Amritsar booklet) and
`Calender_2026-1.pdf` (official 558 wall poster). The package documents its
transcription/page provenance. Attribution and the MIT license are retained
in the script and `SGPC_CALENDAR_LICENSE.txt`. The package/astronomy engine
is not a runtime dependency, and the player makes no calendar network calls.

For example, **13 October 2026 in India is ੨੭ ਅੱਸੂ ੫੫੮**. The fixed 2003
calendar would give a different date and is deliberately not used. Sawan
558 has 32 days. The clock updates every second, and the date changes at
India midnight.

Maintenance: add the next published SGPC Jantri's month starts and advance
`coverageEnd` when it becomes available. Current verified coverage ends
immediately before **15 March 2027**. Outside published coverage the date is
hidden and India time continues; the player does not label astronomical
estimates or fixed-calendar fallbacks as official SGPC dates. The month
starts cannot safely be extrapolated as fixed Gregorian dates.

Run `node scripts/validate_sgpc_calendar.mjs` for date boundaries, India
midnight, the 32-day month, year change and coverage checks. The optional
browser check in `scripts/validate_player_ui.cjs` verifies India time while
its browser is located in Brisbane, and checks desktop and mobile metadata
placement.
