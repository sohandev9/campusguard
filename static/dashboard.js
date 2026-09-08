/*
 * CampusGuard AI — Dashboard Polling Script
 * Polls /stats every 500ms and /alerts every 1s for live updates.
 * Auto-refreshes the alert log and stat cards without page reload.
 */

(function () {
  "use strict";

  // --- DOM references ---
  const el = {
    statusDot:    document.getElementById("statusDot"),
    statusText:   document.getElementById("statusText"),
    currentTime:  document.getElementById("currentTime"),
    statPeople:   document.getElementById("statPeople"),
    statBags:     document.getElementById("statBags"),
    statFight:    document.getElementById("statFight"),
    statFall:     document.getElementById("statFall"),
    statUnattended: document.getElementById("statUnattended"),
    totalFight:   document.getElementById("totalFight"),
    totalFall:    document.getElementById("totalFall"),
    totalUnattended: document.getElementById("totalUnattended"),
    ledPeople:    document.getElementById("ledPeople"),
    ledBags:      document.getElementById("ledBags"),
    ledFight:     document.getElementById("ledFight"),
    ledFall:      document.getElementById("ledFall"),
    ledUnattended: document.getElementById("ledUnattended"),
    logEntries:   document.getElementById("logEntries"),
    alertBadge:   document.getElementById("alertBadge"),
    badgeItems:   document.getElementById("badgeItems"),
    videoPlaceholder: document.getElementById("videoPlaceholder"),
    webcamBtn:    document.getElementById("webcamBtn"),
    uploadBtn:    document.getElementById("uploadBtn"),
    fileInput:    document.getElementById("fileInput"),
    uploadForm:   document.getElementById("uploadForm"),
    filenameDisplay: document.getElementById("filenameDisplay"),
  };

  // --- State ---
  let lastAlertSignature = "";  // dedupe log entries
  let lastStatsHash = "";

  // --- Clock: updates every second ---
  function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, "0");
    const m = String(now.getMinutes()).padStart(2, "0");
    const s = String(now.getSeconds()).padStart(2, "0");
    if (el.currentTime) {
      el.currentTime.textContent = `${h}:${m}:${s}`;
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // --- Health check ---
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.models_loaded) {
        el.statusDot.classList.remove("offline");
        el.statusText.textContent = "SYSTEM ONLINE";
      } else {
        el.statusDot.classList.add("offline");
        el.statusText.textContent = "MODEL LOAD FAILED";
      }
    } catch {
      el.statusDot.classList.add("offline");
      el.statusText.textContent = "DISCONNECTED";
    }
  }
  checkHealth();
  setInterval(checkHealth, 5000);

  // --- Stats polling (500ms) ---
  async function fetchStats() {
    try {
      const res = await fetch("/stats");
      const s = await res.json();
      updateStats(s);
    } catch (err) {
      // Silently skip — connection might be briefly unavailable during stream restarts
    }
  }

  function updateStats(s) {
    // Only update DOM if data changed (avoid unnecessary re-renders)
    const hash = JSON.stringify(s);
    if (hash === lastStatsHash) return;
    lastStatsHash = hash;

    if (el.statPeople)    el.statPeople.textContent    = s.people_tracked || 0;
    if (el.statBags)      el.statBags.textContent      = s.bags_detected || 0;
    if (el.statFight)     el.statFight.textContent     = s.active_fight_alerts || 0;
    if (el.statFall)      el.statFall.textContent      = s.active_fall_alerts || 0;
    if (el.statUnattended) el.statUnattended.textContent = s.unattended_bags || 0;

    if (el.totalFight)    el.totalFight.textContent    = s.total_fight_alerts || 0;
    if (el.totalFall)     el.totalFall.textContent     = s.total_fall_alerts || 0;
    if (el.totalUnattended) el.totalUnattended.textContent = s.total_unattended_bags || 0;

    // LED states
    toggleLed(el.ledPeople,    s.people_tracked > 0);
    toggleLed(el.ledBags,      s.bags_detected > 0);
    toggleLed(el.ledFight,     s.active_fight_alerts > 0);
    toggleLed(el.ledFall,      s.active_fall_alerts > 0);
    toggleLed(el.ledUnattended, s.unattended_bags > 0);

    // Alert badge
    const activeAlerts = (s.active_fight_alerts || 0) + (s.active_fall_alerts || 0) + (s.unattended_bags || 0);
    if (activeAlerts > 0 && el.alertBadge) {
      el.alertBadge.style.display = "block";
      const items = [];
      if (s.active_fight_alerts > 0) items.push(`FIGHT ×${s.active_fight_alerts}`);
      if (s.active_fall_alerts > 0)  items.push(`FALL ×${s.active_fall_alerts}`);
      if (s.unattended_bags > 0)     items.push(`BAG ×${s.unattended_bags}`);
      el.badgeItems.innerHTML = items.map(i => `<span class="badge-item">${i}</span>`).join("");
    } else if (el.alertBadge) {
      el.alertBadge.style.display = "none";
    }

    // Hide video placeholder once we detect people
    if (el.videoPlaceholder && s.people_tracked > 0) {
      el.videoPlaceholder.style.display = "none";
    }
  }

  function toggleLed(ledEl, on) {
    if (!ledEl) return;
    ledEl.classList.toggle("on", on);
  }

  // --- Alert log polling (1000ms) ---
  async function fetchAlerts() {
    try {
      const res = await fetch("/alerts");
      const data = await res.json();
      updateAlertLog(data.alerts || []);
    } catch (err) {
      // skip
    }
  }

  function updateAlertLog(alerts) {
    // Dedupe: only rebuild if the set changed
    const sig = alerts.map(a => a.timestamp + a.type + a.person_id).join("|");
    if (sig === lastAlertSignature) return;
    lastAlertSignature = sig;

    if (!el.logEntries) return;

    if (alerts.length === 0) {
      el.logEntries.innerHTML = '<div class="log-placeholder">No alerts yet. Waiting for detection...</div>';
      return;
    }

    el.logEntries.innerHTML = alerts.map(entry => {
      const isRecent = (Date.now() - new Date(entry.timestamp.replace(/-/g, "/")).getTime()) < 5000;
      return `
        <div class="log-entry ${isRecent ? "recent" : ""}">
          <span class="log-type ${entry.type}">${entry.type}</span>
          <div class="log-meta">
            <span class="log-target">${escapeHtml(entry.person_id || "")}</span>
            <span class="log-time">${entry.timestamp}</span>
          </div>
          ${entry.extra ? `<div class="log-extra">${escapeHtml(entry.extra)}</div>` : ""}
        </div>
      `;
    }).join("");
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // --- Source controls ---
  if (el.webcamBtn) {
    el.webcamBtn.addEventListener("click", () => {
      window.location.href = "/?source=webcam";
    });
  }

  if (el.uploadBtn && el.fileInput && el.uploadForm) {
    el.uploadBtn.addEventListener("click", () => {
      el.fileInput.click();
    });
    el.fileInput.addEventListener("change", () => {
      const file = el.fileInput.files[0];
      if (file) {
        el.filenameDisplay.textContent = file.name;
        // Auto-submit the native form — server redirects to /?source=<unique_name>
        el.uploadForm.submit();
      } else {
        el.filenameDisplay.textContent = "No file selected";
      }
    });
  }

  // --- Start polling ---
  setInterval(fetchStats, 500);
  setInterval(fetchAlerts, 1000);
  fetchStats();
  fetchAlerts();

})();
