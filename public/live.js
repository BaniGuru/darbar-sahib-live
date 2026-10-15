import { darbarDateTime } from "/static/sgpc-calendar.js?v=18";
const $ = id => document.getElementById(id);
let audioUrl = "";
const debugMode = new URLSearchParams(location.search).has("debug");
const audioSessionId = crypto.randomUUID
  ? crypto.randomUUID()
  : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === "x" ? r : (r & 3 | 8)).toString(16);
    });
let playbackBase = null;
let clientStartedAt = null;
let stateRevision = "";
let refreshInFlight = false;
let dockHideTimer = null;
let dockWakeTimer = null;
let playerIsPlaying = false;
let hasStartedPlayback = false;
let playbackLoading = false;
let latestState = null;
let preparedPlaybackState = null;
let panktiBoundaryTimer = null;
let renderedPanktiKey = "";
let wantsAudioPlayback = false;
let audioRecoveryTimer = null;
let audioRecoveryRunning = false;
let audioRecoveryAttempts = 0;
let audioStallTimer = null;
let playbackDelaySeconds = 10;
let audioGeneration = 0;
const maxAudioRecoveryAttempts = 6;
const localTimeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit", minute: "2-digit", hourCycle: "h12"
});

function showPlayerDock() {
  const dock = document.querySelector(".player-dock");
  dock.classList.remove("is-hidden");
  $("playToggle").focus({ preventScroll: true });
  clearTimeout(dockHideTimer);
  dockHideTimer = null;
  if (playerIsPlaying && !playbackLoading) {
    dockHideTimer = setTimeout(() => dock.classList.add("is-hidden"), 5000);
  }
}

function schedulePlayerWake() {
  clearTimeout(dockWakeTimer);
  const now = new Date();
  const nextWake = new Date(now);
  nextWake.setSeconds(0, 0);
  nextWake.setMinutes(Math.floor(now.getMinutes() / 5) * 5 + 5);
  dockWakeTimer = setTimeout(() => {
    showPlayerDock();
    schedulePlayerWake();
  }, Math.max(0, nextWake.getTime() - now.getTime()));
}

function clearAudioRecovery() {
  clearTimeout(audioRecoveryTimer);
  audioRecoveryTimer = null;
  clearTimeout(audioStallTimer);
  audioStallTimer = null;
}

function failAudioRecovery() {
  wantsAudioPlayback = false;
  playerIsPlaying = false;
  clearAudioRecovery();
  setPlaybackLoading(false);
  setPlayerState(false);
  $("liveStatus").textContent = "ਆਵਾਜ਼ ਰੁਕ ਗਈ ਹੈ — ਮੁੜ ਚਲਾਉਣ ਲਈ ਦਬਾਓ";
}

function scheduleAudioRecovery() {
  if (!wantsAudioPlayback) return;
  setPlaybackLoading(true);
  if (audioRecoveryTimer || audioRecoveryRunning) return;
  if (audioRecoveryAttempts >= maxAudioRecoveryAttempts) {
    failAudioRecovery();
    return;
  }

  const delayMs = Math.min(3000 * (2 ** audioRecoveryAttempts), 30000);
  audioRecoveryAttempts += 1;
  $("liveStatus").textContent = "ਆਵਾਜ਼ ਮੁੜ ਜੋੜੀ ਜਾ ਰਹੀ ਹੈ";
  audioRecoveryTimer = setTimeout(async () => {
    audioRecoveryTimer = null;
    if (!wantsAudioPlayback) return;

    const audio = $("liveAudio");
    audioRecoveryRunning = true;
    let attemptTimeout;
    try {
      playbackBase = 0;
      stateRevision = "";
      audio.load();
      await Promise.race([
        audio.play().then(() => true, () => false),
        new Promise(resolve => { attemptTimeout = setTimeout(() => resolve(null), (playbackDelaySeconds + 15) * 1000); })
      ]).then(result => {
        if (result === null && wantsAudioPlayback) audio.pause();
      });
    } catch {
      // The media error and retry path below handle failed reconnect attempts.
    } finally {
      clearTimeout(attemptTimeout);
      audioRecoveryRunning = false;
    }

    if (wantsAudioPlayback && audio.paused) scheduleAudioRecovery();
  }, delayMs);
}

function updateLocalTime() {
  const now = new Date();
  const clock = $("currentUserTime");
  const waiting = hasStartedPlayback && Boolean(latestState?.waiting_for_shabad);
  if (playerIsPlaying || waiting) {
    clock.textContent = localTimeFormatter.formatToParts(now)
      .filter(part => part.type !== "dayPeriod")
      .map(part => part.value)
      .join("")
      .trim();
  } else {
    clock.textContent = "Start Listening";
  }
  clock.dateTime = now.toISOString();
  const darbar = darbarDateTime(now);
  $("darbarNanakshahiDate").textContent = darbar.date;
  $("darbarNanakshahiDate").hidden = !darbar.date;
  $("darbarIndiaTime").textContent = "ਅੰਮ੍ਰਿਤਸਰ " + darbar.time;
  $("darbarIndiaTime").dateTime = now.toISOString();
}

function setPlayerState(playing) {
  playerIsPlaying = playing;
  updateLocalTime();
  const button = $("playToggle");
  button.setAttribute("aria-pressed", String(playing));
  button.setAttribute("aria-label", playbackLoading ? "ਕੀਰਤਨ ਸ਼ੁਰੂ ਹੋ ਰਿਹਾ ਹੈ" : playing ? "ਲਾਈਵ ਕੀਰਤਨ ਰੋਕੋ" : "ਲਾਈਵ ਕੀਰਤਨ ਚਲਾਓ");
  showPlayerDock();
}

function setPlaybackLoading(loading) {
  playbackLoading = loading;
  const button = $("playToggle");
  button.disabled = loading || !audioUrl;
  button.setAttribute("aria-busy", String(loading));
  document.querySelector(".player-dock").classList.toggle("is-loading", loading);
  button.setAttribute("aria-label", loading ? "ਕੀਰਤਨ ਸ਼ੁਰੂ ਹੋ ਰਿਹਾ ਹੈ" : playerIsPlaying ? "ਲਾਈਵ ਕੀਰਤਨ ਰੋਕੋ" : "ਲਾਈਵ ਕੀਰਤਨ ਚਲਾਓ");
  renderWelcome();
  if (loading) showPlayerDock();
}

function onPlaybackStarted() {
  hasStartedPlayback = true;
  wantsAudioPlayback = true;
  audioRecoveryAttempts = 0;
  clearAudioRecovery();
  setPlaybackLoading(false);
  $("welcomeScreen").hidden = true;
  if (clientStartedAt === null) clientStartedAt = Date.now() / 1000;
  if (playbackBase === null && Number.isFinite($("liveAudio").currentTime)) {
    playbackBase = $("liveAudio").currentTime;
  }
  showPlaybackMatch(preparedPlaybackState || latestState || {});
  showDebugPanel();
  setPlayerState(true);
}

async function togglePlayback() {
  if (playbackLoading || !audioUrl) return;
  const audio = $("liveAudio");
  showPlayerDock();
  if (wantsAudioPlayback && playerIsPlaying) {
    wantsAudioPlayback = false;
    playerIsPlaying = false;
    clearAudioRecovery();
    setPlaybackLoading(false);
    audio.pause();
    setPlayerState(false);
    window.location.reload();
    return;
  }
  wantsAudioPlayback = true;
  audioRecoveryAttempts = 0;
  clearAudioRecovery();
  setPlaybackLoading(true);
  try {
    await audio.play();
    if (!audio.paused) {
      onPlaybackStarted();
    }
  } catch {
    scheduleAudioRecovery();
  }
}

function setText(id, value) {
  const node = $(id);
  node.textContent = value || "";
  node.hidden = !value;
}

function setGurbani(id, value) {
  const node = $(id);
  node.replaceChildren();
  const text = value || "";
  for (const piece of text.split(/(\s+)/u)) {
    if (!piece) continue;
    if (/^\s+$/u.test(piece)) {
      node.append(document.createTextNode(piece));
      continue;
    }
    const punctuation = piece.endsWith(";") ? ";" : /[.,]$/u.test(piece) ? piece.slice(-1) : "";
    if (!punctuation) {
      node.append(document.createTextNode(piece));
      continue;
    }
    const span = document.createElement("span");
    span.className = punctuation === ";" ? "pankti-pause-red" : "pankti-pause-blue";
    span.textContent = piece.slice(0, -1);
    node.append(span);
  }
  node.hidden = !text;
}

function fitPunjabi() {
  for (const id of ["legacyGurmukhi", "punjabi"]) {
    const node = $(id);
    if (node.hidden || !node.textContent) continue;
    const start = id.startsWith("legacyGurmukhi") ? 48 : 34;
    const min = 19;
    const measure = document.createElement("div");
    Object.assign(measure.style, {
      position: "fixed", visibility: "hidden", pointerEvents: "none",
      left: "-10000px", top: "0", width: `${Math.min(window.innerWidth * .95, 1100)}px`,
      fontFamily: "GurbaniWebThick,serif", lineHeight: "1.4", whiteSpace: "normal",
      overflowWrap: "anywhere", letterSpacing: window.innerWidth < 600 ? "-.8px" : "-.4px"
    });
    measure.textContent = node.textContent;
    document.body.append(measure);
    let size = Math.min(start, window.innerWidth * (window.innerWidth < 600 ? .075 : .05));
    for (; size > min; size--) {
      measure.style.fontSize = `${size}px`;
      if (measure.scrollHeight <= size * 2.8) break;
    }
    node.style.fontSize = `${size}px`;
    measure.remove();
  }
}

function renderWelcome() {
  const waiting = hasStartedPlayback && Boolean(latestState?.waiting_for_shabad);
  $("welcomeScreen").hidden = hasStartedPlayback && !waiting;
  $("startingSoon").hidden = !waiting && (hasStartedPlayback || !playbackLoading);
  $("startingSoon").textContent = waiting ? "Waiting for next shabad…" : "Starting soon…";
}

function showPlaybackMatch(state) {
  preparedPlaybackState = state;
  syncPanktiToPlayback();
}

function syncPanktiToPlayback() {
  clearTimeout(panktiBoundaryTimer);
  panktiBoundaryTimer = null;
  if (!preparedPlaybackState) return;
  const state = preparedPlaybackState;
  const audio = $("liveAudio");
  let effective = state;
  let nextBoundary = null;
  let playhead = null;
  if (hasStartedPlayback && Number.isFinite(state.audio_start_timestamp)) {
    playhead = state.audio_start_timestamp + Math.max(0, audio.currentTime - (playbackBase || 0));
    for (const entry of state.pankti_schedule || []) {
      if (!Number.isFinite(entry.audio_timestamp) || !entry.line) continue;
      if (entry.audio_timestamp > playhead) {
        nextBoundary = entry.audio_timestamp;
        break;
      }
      const line = entry.line;
      effective = {...state, ...line, pankti: line.punjabi || "",
        pankti_timestamp: entry.audio_timestamp,
        waiting_for_shabad: Boolean(line.waiting_for_shabad)};
    }
  }
  const key = JSON.stringify([hasStartedPlayback, effective.pankti_id,
    effective.pankti_timestamp, effective.waiting_for_shabad,
    effective.legacy_gurmukhi, effective.punjabi, effective.english]);
  latestState = effective;
  if (key !== renderedPanktiKey) {
    renderedPanktiKey = key;
    showMatch(effective);
  }
  if (nextBoundary !== null && wantsAudioPlayback && !audio.paused) {
    // Recheck the actual audio clock on wake-up; buffering must not advance text.
    panktiBoundaryTimer = setTimeout(syncPanktiToPlayback,
      Math.max(20, (nextBoundary - playhead) * 1000 / (audio.playbackRate || 1)));
  }
}

for (const event of ["timeupdate", "seeked", "ratechange", "pause"]) {
  $("liveAudio").addEventListener(event, syncPanktiToPlayback);
}

function showMatch(state) {
  latestState = state;
  renderWelcome();
  if (!hasStartedPlayback || state.waiting_for_shabad) {
    $("panktiGroup").hidden = true;
    $("emptyText").hidden = true;
    return;
  }
  const hasMatch = Boolean(state.punjabi || state.pankti);
  $("panktiGroup").hidden = !hasMatch;
  $("emptyText").hidden = hasMatch;
  if (!hasMatch) {
    $("emptyText").textContent = "ਵਾਹਿਗੁਰੂ";
    for (const id of ["legacyGurmukhi", "punjabi", "english"]) $(id).textContent = "";
    return;
  }
  // The server selects the one confirmed line at this audio position.
  setGurbani("legacyGurmukhi", state.legacy_gurmukhi);
  setGurbani("punjabi", state.punjabi || state.pankti);
  setText("english", state.english);
  fitPunjabi();
}

function showDebugPanel() {
  if (!debugMode) return;
  $("debugPanel").hidden = !hasStartedPlayback;
  // Debug captions follow the same delayed playback clock as the pankti.
  // Recognition diagnostics remain available through /api/live/debug.
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const audio = $("liveAudio");
    if (playbackBase === null) playbackBase = 0;
    const playedSeconds = playbackBase === null
      ? 0
      : Math.max(0, audio.currentTime - playbackBase);
    const query = new URLSearchParams({
      session_id: audioSessionId,
      played_seconds: String(playedSeconds)
    });
    if (clientStartedAt !== null) query.set("client_started_at", String(clientStartedAt));
    if (stateRevision) query.set("known_revision", stateRevision);
    const response = await fetch(`/api/live/state?${query}`, { cache: "no-store" });
    if (response.status === 204) return;
    const state = await response.json();
    if (!response.ok) throw new Error(state.detail || "state unavailable");
    stateRevision = state.revision || "";
    const nextGeneration = Number(state.audio_generation || 0);
    if (audioGeneration && nextGeneration !== audioGeneration) {
      playbackBase = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
      stateRevision = "";
    }
    audioGeneration = nextGeneration;
    playbackDelaySeconds = Number(state.playback_delay_seconds ?? 10);
    if (typeof state.listener_count === "number" && Number.isFinite(state.listener_count)) {
      const count = Math.max(0, Math.floor(state.listener_count));
      $("listenerCount").textContent = new Intl.NumberFormat().format(count);
      $("listenerLabel").textContent = count === 1 ? "listener" : "listeners";
    }
    $("liveDot").classList.toggle("active", Boolean(state.connected));
    $("liveStatus").textContent = !state.playback_ready ? "ਆਵਾਜ਼ ਤਿਆਰ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ" : state.connected ? "ਲਾਈਵ" : "ਜੋੜਿਆ ਜਾ ਰਿਹਾ ਹੈ";
    if (debugMode) {
      $("debugFinal").textContent = state.playback_text || "";
      $("debugPartial").textContent = "";
      $("debugStatus").textContent = "playback";
    }
    if (state.audio_url && state.audio_url !== audioUrl) {
      const needsFocus = $("playToggle").disabled;
      audioUrl = state.audio_url;
      playbackBase = 0;
      audio.src = audioUrl;
      $("playToggle").disabled = playbackLoading;
      if (needsFocus) showPlayerDock();
    }
    showPlaybackMatch(state);
  } catch {
    $("liveDot").classList.remove("active");
    $("liveStatus").textContent = "ਲਾਈਵ ਸਥਿਤੀ ਉਪਲਬਧ ਨਹੀਂ";
  } finally {
    refreshInFlight = false;
  }
}

window.addEventListener("resize", fitPunjabi);
document.fonts?.ready?.then(fitPunjabi);
window.addEventListener("pointermove", event => {
  if (event.pointerType === "mouse" || event.pointerType === "pen") showPlayerDock();
}, { passive: true });
window.addEventListener("touchstart", showPlayerDock, { passive: true });
window.addEventListener("pointerdown", showPlayerDock, { passive: true });
window.addEventListener("keydown", event => {
  const dock = document.querySelector(".player-dock");
  if (event.key === "Enter" && dock.classList.contains("is-hidden")) {
    event.preventDefault();
    showPlayerDock();
    return;
  }
  showPlayerDock();
}, true);
$("playToggle").addEventListener("click", togglePlayback);
$("liveAudio").addEventListener("playing", onPlaybackStarted);
$("liveAudio").addEventListener("error", () => {
  if (wantsAudioPlayback) scheduleAudioRecovery();
  else {
    setPlaybackLoading(false);
    setPlayerState(false);
    $("liveStatus").textContent = "ਆਵਾਜ਼ ਚਲਾਉਣ ਲਈ ਮੁੜ ਕੋਸ਼ਿਸ਼ ਕਰੋ";
  }
});
$("liveAudio").addEventListener("pause", () => {
  if (wantsAudioPlayback) scheduleAudioRecovery();
  else setPlayerState(false);
});
$("liveAudio").addEventListener("ended", () => {
  if (wantsAudioPlayback) scheduleAudioRecovery();
});
$("liveAudio").addEventListener("stalled", () => {
  clearTimeout(audioStallTimer);
  audioStallTimer = setTimeout(() => {
    const audio = $("liveAudio");
    if (!wantsAudioPlayback || audio.readyState >= 3) return;
    if (!audio.paused) audio.pause();
    scheduleAudioRecovery();
  }, (playbackDelaySeconds + 15) * 1000);
});
updateLocalTime();
setInterval(updateLocalTime, 1000);
showPlayerDock();
schedulePlayerWake();
refresh();
setInterval(refresh, 1000);
showDebugPanel();
