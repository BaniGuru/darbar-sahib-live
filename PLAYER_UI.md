# Welcome artwork and playback feedback

The welcome screen uses the supplied `public/darbar-sahib-live.png` without
modifying its pixels. It shows the current listener count below the lower
right edge of the artwork. Gurbani and the optional debug transcript remain
hidden until audio actually starts.

Pressing play disables the button and replaces its icon with an animated
spinner. “Starting soon…” appears below the image. Repeated clicks cannot
start another play request. When playback starts, the welcome screen gives
way to the existing Gurbani view. Automatic pause/recovery keeps that view
and shows loading feedback on the button; manual stop retains the existing
reload behavior and returns to the welcome screen.

`/api/live/state` exposes `listener_count`, counting connected public audio
streams, including clients buffering audio. The count participates in the
state revision, so a waiting visitor sees changes even when the pankti is
unchanged. Reconnecting the same audio session does not count twice.
Recognition, KenLM, matching and the 10-second playback delay are unchanged.

Checks passed: all 110 Python regressions, desktop/mobile browser checks,
and an actual AAC playback/stop check with another tab observing the count
change from zero to one and back to zero. See
[validation results](validation/player-ui/summary.json) and
[live browser check](validation/player-ui/live-browser.json).

Optional browser regressions require a running application, Node, Chromium
and Playwright. Install the test tool outside the checkout:

```bash
npm --cache /tmp/darbar-npm-cache --prefix /tmp/darbar-ui-tools install playwright --no-audit --no-fund
PLAYER_UI_TOOLS_DIR=/tmp/darbar-ui-tools node scripts/validate_player_ui.cjs
```

The browser test uses controlled media/state fixtures to exercise pending
play, duplicate clicks, starting playback, automatic recovery, fresh visits,
listener formatting and responsive layout. It verifies the real PNG asset
loads. `PLAYER_UI_URL`, `PLAYER_UI_OUTPUT` and `CHROMIUM_PATH` optionally
configure the server, artifacts and browser executable. Playwright is a
development test tool and is not an application runtime dependency.
