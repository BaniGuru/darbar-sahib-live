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
let wantsAudioPlayback = false;
let audioRecoveryTimer = null;
let audioRecoveryRunning = false;
let audioRecoveryAttempts = 0;
let audioStallTimer = null;
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
  if (playerIsPlaying) {
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
  $("playToggle").disabled = false;
  setPlayerState(false);
  $("liveStatus").textContent = "ਆਵਾਜ਼ ਰੁਕ ਗਈ ਹੈ — ਮੁੜ ਚਲਾਉਣ ਲਈ ਦਬਾਓ";
}

function scheduleAudioRecovery() {
  if (!wantsAudioPlayback || audioRecoveryTimer || audioRecoveryRunning) return;
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
      audio.load();
      await Promise.race([
        audio.play().then(() => true, () => false),
        new Promise(resolve => { attemptTimeout = setTimeout(() => resolve(null), 15000); })
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
  clock.textContent = localTimeFormatter.formatToParts(now)
    .filter(part => part.type !== "dayPeriod")
    .map(part => part.value)
    .join("")
    .trim();
  clock.dateTime = now.toISOString();
}

function setPlayerState(playing) {
  playerIsPlaying = playing;
  const button = $("playToggle");
  button.setAttribute("aria-pressed", String(playing));
  button.setAttribute("aria-label", playing ? "ਲਾਈਵ ਕੀਰਤਨ ਰੋਕੋ" : "ਲਾਈਵ ਕੀਰਤਨ ਚਲਾਓ");
  showPlayerDock();
}

async function togglePlayback() {
  const audio = $("liveAudio");
  showPlayerDock();
  if (wantsAudioPlayback && playerIsPlaying) {
    wantsAudioPlayback = false;
    playerIsPlaying = false;
    clearAudioRecovery();
    audio.pause();
    setPlayerState(false);
    window.location.reload();
    return;
  }
  wantsAudioPlayback = true;
  audioRecoveryAttempts = 0;
  clearAudioRecovery();
  $("playToggle").disabled = true;
  try {
    await audio.play();
    if (!audio.paused) {
      $("playToggle").disabled = false;
      setPlayerState(true);
    }
  } catch {
    $("playToggle").disabled = false;
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
    const start = id === "legacyGurmukhi" ? 48 : 34;
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

function showMatch(state) {
  const hasMatch = Boolean(state.punjabi || state.pankti);
  $("panktiGroup").hidden = !hasMatch;
  $("emptyText").hidden = hasMatch;
  if (!hasMatch) {
    $("emptyText").textContent = "ਵਾਹਿਗੁਰੂ";
    for (const id of ["legacyGurmukhi", "punjabi", "english"]) $(id).textContent = "";
    return;
  }
  setGurbani("legacyGurmukhi", state.legacy_gurmukhi);
  setGurbani("punjabi", state.punjabi || state.pankti);
  setText("english", state.english);
  fitPunjabi();
}

function connectDebugSocket() {
  if (!debugMode) return;
  $("debugPanel").hidden = false;
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${scheme}//${location.host}/api/live/debug`);
  socket.addEventListener("open", () => { $("debugStatus").textContent = "live"; });
  socket.addEventListener("message", event => {
    try {
      const message = JSON.parse(event.data);
      if (message.type !== "transcript") return;
      const finalWords = Array.isArray(message.final_words)
        ? message.final_words.slice(-30)
        : String(message.final_text || "").split(/\s+/).filter(Boolean).slice(-30);
      const partialWords = Array.isArray(message.partial_words)
        ? message.partial_words
        : String(message.partial_text || "").split(/\s+/).filter(Boolean);
      $("debugFinal").textContent = finalWords.length ? `${finalWords.join(" ")} ` : "";
      $("debugPartial").textContent = partialWords.join(" ");
      $("debugStatus").textContent = "live";
    } catch {
      // Ignore malformed or unrelated messages.
    }
  });
  socket.addEventListener("close", () => {
    $("debugStatus").textContent = "reconnecting…";
    setTimeout(connectDebugSocket, 2000);
  });
  socket.addEventListener("error", () => socket.close());
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const audio = $("liveAudio");
    if (playbackBase === null && Number.isFinite(audio.currentTime) && audio.currentTime > 0) {
      playbackBase = audio.currentTime;
    }
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
    $("liveDot").classList.toggle("active", Boolean(state.connected));
    $("liveStatus").textContent = state.connected ? "ਲਾਈਵ" : "ਜੋੜਿਆ ਜਾ ਰਿਹਾ ਹੈ";
    if (state.audio_url && state.audio_url !== audioUrl) {
      const needsFocus = $("playToggle").disabled;
      audioUrl = state.audio_url;
      audio.src = audioUrl;
      $("playToggle").disabled = false;
      if (needsFocus) showPlayerDock();
    }
    showMatch(state);
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
$("liveAudio").addEventListener("playing", () => {
  wantsAudioPlayback = true;
  audioRecoveryAttempts = 0;
  clearAudioRecovery();
  $("playToggle").disabled = false;
  if (clientStartedAt === null) clientStartedAt = Date.now() / 1000;
  if (playbackBase === null && Number.isFinite($("liveAudio").currentTime)) {
    playbackBase = $("liveAudio").currentTime;
  }
  setPlayerState(true);
});
$("liveAudio").addEventListener("error", () => {
  $("playToggle").disabled = false;
  if (wantsAudioPlayback) scheduleAudioRecovery();
  else {
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
  }, 12000);
});
updateLocalTime();
setInterval(updateLocalTime, 1000);
showPlayerDock();
schedulePlayerWake();
refresh();
setInterval(refresh, 1000);
connectDebugSocket();
