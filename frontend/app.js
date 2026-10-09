const $ = (id) => document.getElementById(id);

/* ===== State ===== */
let videoId = null;
let ws = null;
let alertCount = 0;

/* ===== DOM references ===== */
const videoFeed      = $("videoFeed");
const placeholder    = $("videoPlaceholder");
const videoInput     = $("videoInput");
const filename       = $("filename");
const startBtn       = $("startBtn");
const stopBtn        = $("stopBtn");
const alertList      = $("alertList");
const eventCount     = $("eventCount");
const videoAlert     = $("videoAlert");
const videoAlertText = $("videoAlertText");
const toast          = $("toast");

/* ===== Alert type mapping ===== */
/* Maps backend alert.type to status row data-module */
const moduleMap = {
  fall:  "fall",
  fight: "fight",
  bag:   "bags",
  error: "error",
};

/* Display titles per alert type */
const titleMap = {
  fall:    "FALL DETECTED",
  fight:   "FIGHT DETECTED",
  bag:     "UNATTENDED BAG",
  error:   "SYSTEM ERROR",
  tracker: "ALERT",
};

/* ===== Utility ===== */
function showToast(message) {
  toast.textContent = message;
  toast.style.display = "block";
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => (toast.style.display = "none"), 2800);
}

function setConnection(kind, state) {
  const dot   = $(`${kind}Dot`);
  const label = $(`${kind}State`);
  if (!dot || !label) return;
  label.textContent = state;
  dot.classList.toggle("connected", state === "CONNECTED");
  dot.classList.toggle("error",   state === "DISCONNECTED" || state === "ERROR");
}

function currentTime() {
  return new Date().toLocaleTimeString("en-GB", { hour12: false });
}

function formatAlertTime(isoString) {
  if (!isoString) return currentTime();
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return isoString;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/* ===== Session Reset ===== */
function resetSession() {
  /* Clear all alert cards */
  alertList.innerHTML = "";

  /* Restore empty state */
  const empty = document.createElement("div");
  empty.id = "emptyAlerts";
  empty.className = "empty-alerts";
  empty.innerHTML = '<span>NO ACTIVE ALERTS</span><small>Monitoring events will appear here.</small>';
  alertList.appendChild(empty);

  /* Reset event count */
  alertCount = 0;
  eventCount.textContent = "0 EVENTS";

  /* Reset all status rows */
  document.querySelectorAll(".status-row").forEach((row) => {
    const module = row.dataset.module;
    const value  = row.querySelector(".status-value b");
    const icon   = row.querySelector(".status-value i");

    row.classList.remove("alert-fall", "alert-fight", "alert-bags", "alert-error");

    if (module === "tracker") {
      value.textContent = "ACTIVE";
      icon.style.color = "var(--green)";
    } else {
      value.textContent = "READY";
      icon.style.color = "";
    }
  });

  /* Hide video alert overlay */
  if (videoAlert) videoAlert.style.display = "none";

  showToast("New monitoring session started.");
}

/* ===== Alert Display ===== */
function addAlert(payload) {
  const type   = payload?.type || "unknown";
  const detail = payload?.detail || "";
  const ts     = payload?.timestamp;

  const module = moduleMap[type] || "tracker";

  /* Remove empty placeholder */
  const emptyMsg = $("emptyAlerts");
  if (emptyMsg) emptyMsg.remove();

  /* Create alert card */
  const card = document.createElement("div");
  card.className = `alert-card ${module}`;
  card.innerHTML = `
    <div class="alert-top">
      <span class="alert-type">${escapeHtml(titleMap[module] || "ALERT")}</span>
      <span class="alert-time">${escapeHtml(formatAlertTime(ts))}</span>
    </div>
    <div class="alert-detail">${escapeHtml(detail)}</div>
    ${payload.confidence !== undefined && payload.confidence !== null
      ? `<div class="alert-confidence">CONF: ${Number(payload.confidence).toFixed(2)}</div>`
      : ''
    }
  `;

  /* Prepend — newest on top */
  alertList.prepend(card);

  /* Update event count */
  alertCount += 1;
  eventCount.textContent = `${alertCount} EVENTS`;

  /* Flash status chip & video overlay */
  flashModule(module, titleMap[module] || "ALERT", detail);
}

function flashModule(module, title, detail) {
  let row;
  if (module === "error") {
    row = document.querySelector('.status-row[data-module="tracker"]');
  } else {
    row = document.querySelector(`.status-row[data-module="${module}"]`);
  }
  if (!row) return;

  row.classList.remove("alert-fall", "alert-fight", "alert-bags", "alert-error");
  if (module === "fall")  row.classList.add("alert-fall");
  if (module === "fight") row.classList.add("alert-fight");
  if (module === "bags")  row.classList.add("alert-bags");
  if (module === "error") row.classList.add("alert-error");

  const value = row.querySelector(".status-value b");
  const icon  = row.querySelector(".status-value i");
  const old = value.textContent;
  value.textContent = "ALERT";
  icon.style.color = module === "error" ? "var(--red)" : "";

  setTimeout(() => {
    row.classList.remove("alert-fall", "alert-fight", "alert-bags", "alert-error");
    value.textContent = old;
    icon.style.color = "";
  }, 3000);

  /* Video overlay flash */
  if (videoAlert && videoAlertText) {
    videoAlertText.textContent = `${title} • ${detail}`;
    videoAlert.style.display = "block";
    setTimeout(() => (videoAlert.style.display = "none"), 3000);
  }
}

/* ===== WebSocket ===== */
function connectWebSocket() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${location.host}/ws/alerts`);

  ws.onopen = () => {
    setConnection("ws", "CONNECTED");
  };

  ws.onmessage = (event) => {
    try {
      const alert = JSON.parse(event.data);
      addAlert(alert);
    } catch (e) {
      console.error("[WS] parse error:", e);
    }
  };

  ws.onclose = () => {
    setConnection("ws", "ERROR");
    setTimeout(connectWebSocket, 1800);
  };

  ws.onerror = () => {
    setConnection("ws", "ERROR");
  };
}

/* ===== API Calls ===== */
async function uploadVideo() {
  const file = videoInput.files?.[0];
  if (!file) {
    showToast("Choose a video first.");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    const data = await res.json();
    videoId = data.video_id ?? data.id;
    const file = videoInput.files?.[0];
    const displayName = file ? file.name : "uploaded video";
    filename.textContent = `Loaded: ${displayName}`;
    showToast("Video uploaded. Ready to monitor.");
  } catch (err) {
    showToast(err.message);
  }
}

async function startMonitoring() {
  if (!videoId) {
    await uploadVideo();
    if (!videoId) return;
  }

  try {
    const res = await fetch(`/start/${encodeURIComponent(videoId)}`, { method: "POST" });
    if (!res.ok) throw new Error(`Start failed (${res.status})`);

    /* Reset the session: clear alerts, reset chips, reset count */
    resetSession();

    /* Start the MJPEG stream */
    videoFeed.src = `/stream/${encodeURIComponent(videoId)}?t=${Date.now()}`;
    videoFeed.style.display = "block";
    placeholder.style.display = "none";
    setConnection("stream", "CONNECTED");
  } catch (err) {
    showToast(err.message);
    setConnection("stream", "ERROR");
  }
}

async function stopMonitoring() {
  try {
    const res = await fetch("/stop", { method: "POST" });
    if (!res.ok) throw new Error(`Stop failed (${res.status})`);

    videoFeed.removeAttribute("src");
    videoFeed.style.display = "none";
    placeholder.style.display = "grid";
    setConnection("stream", "DISCONNECTED");
    showToast("Monitoring stopped.");
  } catch (err) {
    showToast(err.message);
  }
}

/* ===== Init ===== */
videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  videoId = null;
  if (file) {
    filename.textContent = `Selected: ${file.name}`;
  } else {
    filename.textContent = "No video selected";
  }
});

startBtn.addEventListener("click", startMonitoring);
stopBtn.addEventListener("click", stopMonitoring);

connectWebSocket();
