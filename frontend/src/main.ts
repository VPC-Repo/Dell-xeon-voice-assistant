/**
 * Browser client for the voice assistant.
 *
 * Responsibilities only: microphone capture, WebRTC setup, playback, and
 * rendering server events. All VAD, transcription, generation, and TTS run
 * on the server; this file must stay free of model logic.
 *
 * UI model: a single pulsating orb (colored by pipeline state, scaled by
 * live audio level), a three-button bar (mute / connect / settings), and a
 * settings sheet holding devices, voice, optimization, and the transcript
 * toggle. The transcript itself is opt-in.
 */

type ServerEvent = {
  type: string;
  state?: string;
  text?: string;
  message?: string;
  profile?: string;
  eos_to_first_audio_ms?: number;
  dropped_frames?: number;
};

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const ui = {
  orbTap: $("orb-tap") as HTMLButtonElement,
  orb: $("orb") as HTMLDivElement,
  orbLabel: $("orb-label") as HTMLSpanElement,
  status: $("status") as HTMLDivElement,
  statusDetail: $("status-detail") as HTMLDivElement,
  stateEyebrow: $("state-eyebrow") as HTMLDivElement,
  muteBtn: $("mute") as HTMLButtonElement,
  endBtn: $("end-call") as HTMLButtonElement,
  settingsBtn: $("settings-btn") as HTMLButtonElement,
  sheetClose: $("sheet-close") as HTMLButtonElement,
  backdrop: $("backdrop") as HTMLDivElement,
  micSelect: $("mic-select") as HTMLSelectElement,
  voiceSelect: $("voice-select") as HTMLSelectElement,
  transcript: $("transcript") as HTMLDivElement,
  showTranscript: $("show-transcript") as HTMLInputElement,
  devicesInfo: $("devices-info") as HTMLDivElement,
  connInfo: $("conn-info") as HTMLDivElement,
  toast: $("toast") as HTMLDivElement,
  player: $("player") as HTMLAudioElement,
  perf: $("perf") as HTMLInputElement,
  perfName: $("perf-name") as HTMLElement,
};

const PROFILES = ["stable", "balanced", "fast"] as const;

const PROFILE_LABEL: Record<string, string> = {
  stable: "resilient",
  balanced: "balanced",
  fast: "responsive",
};

const STATUS_TEXT: Record<string, string> = {
  disconnected: "Ready when you are",
  connecting: "Connecting",
  listening: "Listening",
  transcribing: "Understanding",
  thinking: "Thinking",
  speaking: "Speaking",
};

const STATUS_DETAIL: Record<string, string> = {
  disconnected: "",
  connecting: "Securing your private audio session…",
  listening: "Speak naturally. You can interrupt at any time.",
  transcribing: "Making sure I heard you correctly…",
  thinking: "Working through your question locally…",
  speaking: "Start talking at any time to interrupt.",
};

const STATE_EYEBROW: Record<string, string> = {
  disconnected: "CPU voice assistant",
  connecting: "One moment",
  listening: "Ready for your question",
  transcribing: "I heard you",
  thinking: "I heard you",
  speaking: "Responding",
};

function currentProfile(): string {
  return PROFILES[Number(ui.perf.value)] ?? "balanced";
}

function showProfile(name: string): void {
  const idx = PROFILES.indexOf(name as (typeof PROFILES)[number]);
  if (idx >= 0) ui.perf.value = String(idx);
  ui.perfName.textContent = PROFILE_LABEL[name] ?? name;
}

function prettyVoice(id: string): string {
  const cap = (s: string) =>
    s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  // "kokoro-af_heart" -> "Heart (Kokoro US F)"
  const k = id.match(/^kokoro-([ab])([fm])_(.+)$/);
  if (k) {
    const region = k[1] === "a" ? "US" : "GB";
    return `${cap(k[3])} (Kokoro ${region} ${k[2].toUpperCase()})`;
  }
  return cap(id);
}

async function loadVoices(): Promise<void> {
  try {
    const resp = await api("/api/voices");
    if (!resp.ok) return;
    const body = (await resp.json()) as { voices: string[]; default: string };
    ui.voiceSelect.innerHTML = "";
    if (!body.voices.length) return;
    $("voice-row").hidden = false;
    for (const v of body.voices) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = prettyVoice(v);
      ui.voiceSelect.appendChild(opt);
    }
    const saved = localStorage.getItem("va_voice");
    ui.voiceSelect.value =
      saved && body.voices.includes(saved) ? saved : body.default;
  } catch {
    /* voices row stays empty */
  }
}

let pc: RTCPeerConnection | null = null;
let channel: RTCDataChannel | null = null;
let micStream: MediaStream | null = null;
let sessionId: string | null = null;
let muted = false;
let pendingAssistantEl: HTMLDivElement | null = null;

// ---- browser-clock latency + transport quality -----------------------------
// The truthful user metric is: last voiced mic sample -> first played
// assistant sample, on THIS clock. Server metrics stop at "PCM pushed".
let audioCtx: AudioContext | null = null;
let lastLocalVoiceAt = 0;
let awaitingReply = false;
let statsTimer: number | null = null;
let rtpReceiver: (RTCRtpReceiver & { playoutDelayHint?: number }) | null = null;
let hintRaised = false;
let lastE2eMs = 0;
let lastServerFirstAudioMs = 0;
const netStats = { rtt: -1, jitterMs: -1, jbufMs: -1, lost: 0, cand: "", relay: false };

// ---- orb ------------------------------------------------------------------
// monitorLevels() writes raw RMS here; one rAF loop smooths it (fast attack,
// slow decay) into the CSS --level var that scales the orb.
const levels = { local: 0, remote: 0 };
let orbLevel = 0;
let orbWritten = -1;
let orbRunning = false;

function orbLoop(): void {
  if (!pc) {
    // Disconnected: park the orb and stop the loop so the idle page is
    // fully quiescent (continuous rAF/style churn costs phone battery).
    orbRunning = false;
    orbLevel = 0;
    orbWritten = 0;
    document.body.style.setProperty("--level", "0");
    document.body.style.setProperty("--orb-scale", "1");
    document.body.style.setProperty("--halo-opacity", "0.68");
    return;
  }
  const state = document.body.dataset.state ?? "disconnected";
  let target = 0;
  if (state === "listening" && !muted) target = Math.min(1, levels.local * 9);
  else if (state === "speaking") target = Math.min(1, levels.remote * 7);
  orbLevel += (target - orbLevel) * (target > orbLevel ? 0.4 : 0.08);
  if (orbLevel < 0.004) orbLevel = 0;
  const q = Math.round(orbLevel * 100) / 100;
  if (q !== orbWritten) {
    document.body.style.setProperty("--level", String(q));
    document.body.style.setProperty("--orb-scale", String(1 + q * 0.035));
    document.body.style.setProperty("--halo-opacity", String(0.68 + q * 0.32));
    orbWritten = q;
  }
  requestAnimationFrame(orbLoop);
}

function startOrb(): void {
  if (!orbRunning) {
    orbRunning = true;
    requestAnimationFrame(orbLoop);
  }
}

function monitorLevels(stream: MediaStream, isLocal: boolean): void {
  if (!audioCtx) return;
  try {
    const src = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const tick = () => {
      if (!pc) {
        levels[isLocal ? "local" : "remote"] = 0;
        return; // disconnected: stop polling
      }
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      levels[isLocal ? "local" : "remote"] = rms;
      if (isLocal) {
        if (rms > 0.02 && !muted) {
          lastLocalVoiceAt = performance.now();
          awaitingReply = true;
        }
      } else if (awaitingReply && rms > 0.01 && lastLocalVoiceAt > 0) {
        awaitingReply = false;
        const e2e = Math.round(performance.now() - lastLocalVoiceAt);
        if (e2e > 150 && e2e < 30000) {
          lastE2eMs = e2e;
          updateConnInfo();
          sendControl({
            type: "client_metrics", e2e_ms: e2e, rtt_ms: netStats.rtt,
            jitter_ms: netStats.jitterMs, jbuf_ms: netStats.jbufMs,
            lost: netStats.lost, cand: netStats.cand, relay: netStats.relay,
          });
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } catch {
    /* WebAudio unavailable (some mobile browsers): server metric only */
  }
}

async function pollStats(): Promise<void> {
  if (!pc) return;
  try {
    const report = await pc.getStats();
    report.forEach((s: any) => {
      if (s.type === "candidate-pair" && (s.selected || s.nominated) && s.state === "succeeded") {
        if (typeof s.currentRoundTripTime === "number") {
          netStats.rtt = Math.round(s.currentRoundTripTime * 1000);
        }
      }
      if (s.type === "remote-candidate" || s.type === "local-candidate") {
        if (s.candidateType === "relay") netStats.relay = true;
      }
      if (s.type === "inbound-rtp" && s.kind === "audio") {
        if (typeof s.jitter === "number") netStats.jitterMs = Math.round(s.jitter * 1000);
        if (s.jitterBufferDelay && s.jitterBufferEmittedCount) {
          netStats.jbufMs = Math.round((s.jitterBufferDelay / s.jitterBufferEmittedCount) * 1000);
        }
        if (typeof s.packetsLost === "number") netStats.lost = s.packetsLost;
      }
    });
    // Adaptive playout: start shallow (set in ontrack); deepen only if the
    // network measurably needs it.
    if (!hintRaised && rtpReceiver && "playoutDelayHint" in rtpReceiver
        && (netStats.jitterMs > 25 || netStats.relay)) {
      rtpReceiver.playoutDelayHint = 0.15;
      hintRaised = true;
    }
    updateConnInfo();
  } catch {
    /* stats unsupported */
  }
}

function updateConnInfo(): void {
  if (!pc) {
    ui.connInfo.textContent = "not connected";
    return;
  }
  const bits: string[] = [];
  if (lastE2eMs > 0) bits.push(`you→voice ${lastE2eMs} ms`);
  if (lastServerFirstAudioMs > 0) bits.push(`first audio ${lastServerFirstAudioMs} ms`);
  if (netStats.rtt >= 0) bits.push(`rtt ${netStats.rtt} ms`);
  if (netStats.jitterMs >= 0) bits.push(`jitter ${netStats.jitterMs} ms`);
  if (netStats.relay) bits.push("via relay");
  ui.connInfo.textContent = bits.length ? bits.join(" · ") : "connected";
}

async function updateDevicesInfo(): Promise<void> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mic =
      micStream?.getAudioTracks()[0]?.label ||
      devices.find((d) => d.kind === "audioinput" && d.label)?.label ||
      "microphone (name hidden until connected)";
    const out =
      devices.find((d) => d.kind === "audiooutput" && d.label)?.label ||
      "system default speaker";
    ui.devicesInfo.textContent = `🎤 ${mic} · 🔊 ${out}`;
  } catch {
    ui.devicesInfo.textContent = "—";
  }
}

function setState(state: string): void {
  document.body.dataset.state = state;
  const isMuted = muted && state === "listening";
  ui.status.textContent = isMuted ? "Microphone muted" : STATUS_TEXT[state] ?? state;
  ui.statusDetail.textContent = isMuted
    ? "Tap the microphone to continue the conversation."
    : STATUS_DETAIL[state] ?? "";
  ui.stateEyebrow.textContent = isMuted
    ? "Session remains connected"
    : STATE_EYEBROW[state] ?? "";
  ui.orbLabel.textContent =
    state === "disconnected" ? "Click to connect"
    : state === "connecting" ? "Connecting…"
    : "";
}

function appendLine(role: "user" | "assistant" | "system", text: string): HTMLDivElement {
  const el = document.createElement("div");
  el.className = `line ${role}`;
  el.textContent = text;
  ui.transcript.appendChild(el);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
  return el;
}

let toastTimer: number | null = null;
function showToast(text: string, isError = false): void {
  ui.toast.textContent = text;
  ui.toast.classList.toggle("err", isError);
  ui.toast.classList.add("show");
  if (toastTimer !== null) clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => ui.toast.classList.remove("show"), 4000);
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("va_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function api(path: string, init: RequestInit = {}): Promise<Response> {
  const resp = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...init.headers },
  });
  if (resp.status === 401) {
    const token = prompt("Access token:");
    if (token) {
      localStorage.setItem("va_token", token);
      return api(path, init);
    }
  }
  return resp;
}

async function listMics(): Promise<void> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  ui.micSelect.innerHTML = "";
  devices
    .filter((d) => d.kind === "audioinput")
    .forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${i + 1}`;
      ui.micSelect.appendChild(opt);
    });
  void updateDevicesInfo();
}

function handleServerEvent(ev: ServerEvent): void {
  switch (ev.type) {
    case "state":
      setState(ev.state ?? "unknown");
      break;
    case "final_transcript":
      appendLine("user", ev.text ?? "");
      pendingAssistantEl = null;
      break;
    case "assistant_text_delta":
      if (!pendingAssistantEl) pendingAssistantEl = appendLine("assistant", "");
      pendingAssistantEl.textContent += ev.text ?? "";
      ui.transcript.scrollTop = ui.transcript.scrollHeight;
      break;
    case "assistant_text_final":
      if (pendingAssistantEl) pendingAssistantEl.textContent = ev.text ?? "";
      pendingAssistantEl = null;
      break;
    case "playback_stopped":
      pendingAssistantEl = null;
      break;
    case "profile":
      if (ev.profile) {
        showProfile(ev.profile);
        localStorage.setItem("va_profile", ev.profile);
      }
      break;
    case "metrics":
      if (ev.eos_to_first_audio_ms !== undefined) {
        lastServerFirstAudioMs = ev.eos_to_first_audio_ms;
        updateConnInfo();
      }
      break;
    case "echo_suppressed":
      appendLine("system", "(heard my own voice — ignored)");
      break;
    case "recoverable_error":
      appendLine("system", `⚠ ${ev.message ?? "recoverable error"}`);
      showToast(ev.message ?? "recoverable error", true);
      break;
    case "fatal_error":
      appendLine("system", `✖ ${ev.message ?? "fatal error"}`);
      showToast(ev.message ?? "fatal error", true);
      void disconnect();
      break;
  }
}

async function waitForIceGathering(conn: RTCPeerConnection): Promise<void> {
  if (conn.iceGatheringState === "complete") return;
  await new Promise<void>((resolve) => {
    const check = () => {
      if (conn.iceGatheringState === "complete") {
        conn.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    conn.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 5000); // cap gathering wait (STUN/TURN can be slow)
  });
}

async function connect(): Promise<void> {
  setState("connecting");
  try {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: ui.micSelect.value ? { exact: ui.micSelect.value } : undefined,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      await listMics(); // labels become visible after permission
    } catch {
      // Listen-only fallback: connect without a microphone.
      micStream = null;
      showToast("Microphone unavailable — listen-only.", true);
    }

    const created = await api("/api/sessions", { method: "POST" });
    if (!created.ok) throw new Error(`session refused (${created.status})`);
    const sessionInfo = await created.json();
    sessionId = sessionInfo.session_id;

    // Initialize before ontrack so incoming audio is monitored on the first call.
    audioCtx = audioCtx ?? new AudioContext();
    void audioCtx.resume();

    // ICE topology (STUN/TURN) is server deployment config.
    pc = new RTCPeerConnection({ iceServers: sessionInfo.ice_servers ?? [] });
    channel = pc.createDataChannel("control", { ordered: true });
    channel.onmessage = (e) => handleServerEvent(JSON.parse(e.data));
    channel.onopen = () => {
      // Re-apply remembered preferences to the new session.
      const saved = localStorage.getItem("va_profile");
      if (saved && saved !== "balanced") sendControl({ type: "set_profile", profile: saved });
      const voice = localStorage.getItem("va_voice");
      if (voice) sendControl({ type: "set_voice", voice });
    };
    if (micStream) {
      micStream.getAudioTracks().forEach((t) => pc!.addTrack(t, micStream!));
    } else {
      pc.addTransceiver("audio", { direction: "recvonly" });
    }
    pc.ontrack = (e) => {
      // Start with a shallow playout buffer; pollStats() deepens it only
      // when measured jitter or a TURN relay demands it.
      rtpReceiver = e.receiver as RTCRtpReceiver & { playoutDelayHint?: number };
      hintRaised = false;
      if ("playoutDelayHint" in rtpReceiver) rtpReceiver.playoutDelayHint = 0.08;
      const remote = e.streams[0] ?? new MediaStream([e.track]);
      ui.player.srcObject = remote;
      void ui.player.play();
      monitorLevels(remote, false);
    };
    pc.onconnectionstatechange = () => {
      if (pc && ["failed", "disconnected", "closed"].includes(pc.connectionState)) {
        showToast(`connection ${pc.connectionState}`, true);
        if (pc.connectionState === "failed") void disconnect();
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIceGathering(pc);

    const answered = await api(`/api/sessions/${sessionId}/offer`, {
      method: "POST",
      body: JSON.stringify({
        sdp: pc.localDescription!.sdp,
        type: pc.localDescription!.type,
      }),
    });
    if (!answered.ok) throw new Error(`offer rejected (${answered.status})`);
    await pc.setRemoteDescription(await answered.json());

    if (micStream) monitorLevels(micStream, true);
    statsTimer = window.setInterval(() => void pollStats(), 5000);

    document.body.classList.add("connected");
    startOrb();
    ui.orbTap.setAttribute("aria-label", "Voice activity");
    ui.endBtn.disabled = false;
    ui.muteBtn.disabled = micStream === null;
    updateConnInfo();
    void updateDevicesInfo();
  } catch (err) {
    showToast((err as Error).message, true);
    await disconnect();
  }
}

async function disconnect(): Promise<void> {
  if (statsTimer !== null) {
    clearInterval(statsTimer);
    statsTimer = null;
  }
  rtpReceiver = null;
  awaitingReply = false;
  channel?.close();
  channel = null;
  pc?.close();
  pc = null;
  micStream?.getTracks().forEach((t) => t.stop());
  micStream = null;
  ui.player.srcObject = null;
  const context = audioCtx;
  audioCtx = null;
  if (context) void context.close().catch(() => undefined);
  if (sessionId) {
    void api(`/api/sessions/${sessionId}`, { method: "DELETE" }).catch(() => undefined);
    sessionId = null;
  }
  muted = false;
  levels.local = 0;
  levels.remote = 0;
  lastE2eMs = 0;
  lastServerFirstAudioMs = 0;
  lastLocalVoiceAt = 0;
  hintRaised = false;
  Object.assign(netStats, { rtt: -1, jitterMs: -1, jbufMs: -1, lost: 0, cand: "", relay: false });
  ui.muteBtn.setAttribute("aria-label", "Mute microphone");
  document.body.classList.remove("connected", "muted-on");
  ui.orbTap.setAttribute("aria-label", "Click to connect");
  ui.endBtn.disabled = true;
  ui.muteBtn.disabled = true;
  updateConnInfo();
  setState("disconnected");
}

function sendControl(msg: object): void {
  if (channel?.readyState === "open") channel.send(JSON.stringify(msg));
}

function toggleConnection(): void {
  // The orb starts a session and becomes a non-destructive activity surface
  // once connected. Ending a live session is always an explicit control.
  if (document.body.dataset.state === "connecting" || pc) return;
  void connect();
}

ui.orbTap.addEventListener("click", toggleConnection);
ui.endBtn.addEventListener("click", () => void disconnect());

ui.muteBtn.addEventListener("click", () => {
  muted = !muted;
  micStream?.getAudioTracks().forEach((t) => (t.enabled = !muted));
  sendControl({ type: muted ? "mute" : "unmute" });
  document.body.classList.toggle("muted-on", muted);
  ui.muteBtn.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
  setState(document.body.dataset.state ?? "disconnected");
});

// ---- settings sheet -------------------------------------------------------

function openSheet(): void {
  document.body.classList.add("sheet-open");
  ui.settingsBtn.setAttribute("aria-expanded", "true");
  void updateDevicesInfo();
  updateConnInfo();
}
function closeSheet(): void {
  document.body.classList.remove("sheet-open");
  ui.settingsBtn.setAttribute("aria-expanded", "false");
}
ui.settingsBtn.addEventListener("click", () =>
  document.body.classList.contains("sheet-open") ? closeSheet() : openSheet(),
);
ui.sheetClose.addEventListener("click", closeSheet);
ui.backdrop.addEventListener("click", closeSheet);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeSheet();
});

ui.perf.addEventListener("input", () => {
  const profile = currentProfile();
  showProfile(profile);
  localStorage.setItem("va_profile", profile);
  sendControl({ type: "set_profile", profile });
});

ui.voiceSelect.addEventListener("change", () => {
  const voice = ui.voiceSelect.value;
  localStorage.setItem("va_voice", voice);
  sendControl({ type: "set_voice", voice });
});

ui.micSelect.addEventListener("change", () => {
  void updateDevicesInfo();
  if (pc) showToast("Mic change applies on next connect");
});

ui.showTranscript.addEventListener("change", () => {
  ui.transcript.hidden = !ui.showTranscript.checked;
  localStorage.setItem("va_transcript", ui.showTranscript.checked ? "1" : "0");
  if (ui.showTranscript.checked) ui.transcript.scrollTop = ui.transcript.scrollHeight;
});

// ---- boot -----------------------------------------------------------------

void listMics();
void loadVoices();
showProfile(localStorage.getItem("va_profile") ?? "balanced");
ui.showTranscript.checked = localStorage.getItem("va_transcript") === "1";
ui.transcript.hidden = !ui.showTranscript.checked;
setState("disconnected");
