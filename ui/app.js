const BACKEND_URL = "http://127.0.0.1:5000";

class DriverAttentionDashboard {
  constructor() {
    this.telemetryTimer = null;
    this.connected = false;
    this.lastBlinkCount = 0;
    this.lastYawnCount = 0;
    this.lastDrowsyCount = 0;
    this.currentStatus = "WAITING";
    this.audioContext = null;
    this.criticalInterval = null;
    this.audioUnlocked = false;

    this.backendStatus = document.getElementById("backend-status");
    this.videoFeed = document.getElementById("video-feed");
    this.feedPlaceholder = document.getElementById("feed-placeholder");
    this.connectButton = document.getElementById("connect-btn");
    this.recalibrateButton = document.getElementById("recalibrate-btn");
    this.cameraSelect = document.getElementById("camera-select");
    this.refreshCamerasButton = document.getElementById("refresh-cameras-btn");
    this.switchCameraButton = document.getElementById("switch-camera-btn");

    this.bindEvents();
    this.loadCameras();
    this.tryConnect();
  }

  bindEvents() {
    this.connectButton.addEventListener("click", async () => {
      await this.unlockAudio();
      this.tryConnect(true);
    });
    this.recalibrateButton.addEventListener("click", () => this.recalibrate());
    this.refreshCamerasButton.addEventListener("click", () => this.loadCameras(true));
    this.switchCameraButton.addEventListener("click", () => this.switchCamera());
    this.videoFeed.addEventListener("load", () => this.showFeed());
    this.videoFeed.addEventListener("error", () => this.handleOffline("Video stream unavailable."));
    window.addEventListener("beforeunload", () => this.stopPolling());
    document.addEventListener("click", () => this.unlockAudio(), { once: true });
    document.addEventListener("keydown", () => this.unlockAudio(), { once: true });
  }

  async tryConnect(fromButton = false) {
    if (fromButton) {
      this.setSyncText("Trying to connect to the backend...");
    }

    try {
      const telemetry = await this.fetchTelemetry();
      this.connected = true;
      this.setOnlineState();
      this.startFeed();
      this.updateTelemetry(telemetry);
      this.startPolling();
    } catch (error) {
      this.handleOffline("Backend not reachable. Start python src/main.py first.");
    }
  }

  startFeed() {
    this.videoFeed.src = `${BACKEND_URL}/video_feed`;
  }

  showFeed() {
    this.videoFeed.style.display = "block";
    this.feedPlaceholder.style.display = "none";
  }

  startPolling() {
    this.stopPolling();
    this.telemetryTimer = window.setInterval(async () => {
      try {
        const telemetry = await this.fetchTelemetry();
        this.connected = true;
        this.setOnlineState();
        this.updateTelemetry(telemetry);
      } catch (error) {
        this.handleOffline("Telemetry lost. Waiting for backend...");
      }
    }, 500);
  }

  stopPolling() {
    if (this.telemetryTimer) {
      window.clearInterval(this.telemetryTimer);
      this.telemetryTimer = null;
    }
  }

  async fetchTelemetry() {
    const response = await fetch(`${BACKEND_URL}/telemetry`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`Telemetry request failed: ${response.status}`);
    }

    const payload = await response.json();
    if (!payload || !payload.status) {
      throw new Error("Telemetry payload missing");
    }

    return payload;
  }

  async recalibrate() {
    try {
      const response = await fetch(`${BACKEND_URL}/recalibrate`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Recalibration failed");
      }
      this.setSyncText("Recalibration request sent.");
    } catch (error) {
      this.handleOffline("Could not reach backend for recalibration.");
    }
  }

  async loadCameras(fromButton = false) {
    try {
      const response = await fetch(`${BACKEND_URL}/cameras`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Camera list request failed");
      }

      const payload = await response.json();
      const cameras = payload.cameras || [];
      const currentValue = this.cameraSelect.value;

      this.cameraSelect.innerHTML = "";
      cameras.forEach((camera) => {
        const option = document.createElement("option");
        option.value = String(camera.index);
        option.textContent = camera.label;
        this.cameraSelect.appendChild(option);
      });

      if (cameras.length === 0) {
        const option = document.createElement("option");
        option.value = "0";
        option.textContent = "No cameras found";
        this.cameraSelect.appendChild(option);
      }

      if ([...this.cameraSelect.options].some((option) => option.value === currentValue)) {
        this.cameraSelect.value = currentValue;
      }

      if (fromButton) {
        this.setSyncText("Camera list refreshed.");
      }
    } catch (error) {
      if (fromButton) {
        this.setSyncText("Could not refresh camera list.");
      }
    }
  }

  async switchCamera() {
    const selectedIndex = this.cameraSelect.value;
    try {
      const response = await fetch(`${BACKEND_URL}/select_camera?index=${encodeURIComponent(selectedIndex)}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error("Camera switch failed");
      }

      this.setSyncText(`Switching to camera ${selectedIndex}...`);
      this.setText("camera-source", `Camera ${selectedIndex}`);
    } catch (error) {
      this.setSyncText("Could not switch camera source.");
    }
  }

  updateTelemetry(data) {
    this.handleEvents(data);
    this.setText("driver-status", data.status);
    this.setText("fps-value", this.formatNumber(data.fps, 1));
    this.setText("integration-mode", data.integration_mode || "-");
    this.setText("pose-label", data.pose_label || "CENTER");
    this.setText("blink-count", data.blink_count ?? 0);
    this.setText("yawn-count", data.yawn_alerts ?? 0);
    this.setText("drowsy-count", data.drowsy_alerts ?? 0);
    this.setText("closed-frames", data.closed_frames ?? 0);
    this.setText("ear-value", this.formatNumber(data.score_value, 3));
    this.setText("mar-value", this.formatNumber(data.mar_value, 3));
    this.setText("ear-threshold", this.formatNumber(data.threshold_value, 3));
    this.setText("mar-threshold", this.formatNumber(data.mar_threshold, 3));
    this.setText("tracking-state", data.face_detected ? "Face locked" : "Searching for face");
    this.setText("camera-source", data.camera_name || `Camera ${data.camera_index ?? 0}`);
    this.setText("telemetry-health", "Streaming");
    this.setText(
      "attention-level",
      data.status === "DROWSY" ? "Immediate alert" : data.status === "WARNING" ? "Warning active" : "Attention stable"
    );
    this.setText("face-lock", data.face_detected ? "Face Locked" : "Face Searching");
    this.setSyncText(`Telemetry updated at ${new Date().toLocaleTimeString()}`);
    if (data.camera_index !== undefined) {
      this.cameraSelect.value = String(data.camera_index);
    }

    this.paintStatePills(data);
    this.updateAlertPresentation(data);
    this.updateCoach(data);
    this.updateAudioState(data.status);
  }

  paintStatePills(data) {
    const backendStatus = this.backendStatus;
    const faceLock = document.getElementById("face-lock");
    const attentionLevel = document.getElementById("attention-level");

    backendStatus.className = "status-pill online";
    backendStatus.textContent = "Backend Online";

    faceLock.className = `mini-pill ${data.face_detected ? "online" : "searching"}`;
    attentionLevel.className = `mini-pill ${this.attentionClass(data.status)}`;
  }

  attentionClass(status) {
    if (status === "DROWSY") {
      return "offline";
    }
    if (status === "WARNING" || status === "CALIBRATING") {
      return "neutral";
    }
    return "online";
  }

  setOnlineState() {
    this.connected = true;
    this.backendStatus.className = "status-pill online";
    this.backendStatus.textContent = "Backend Online";
  }

  handleOffline(message) {
    this.connected = false;
    this.stopPolling();
    this.stopAlertLoops();
    this.backendStatus.className = "status-pill offline";
    this.backendStatus.textContent = "Backend Offline";
    this.videoFeed.style.display = "none";
    this.feedPlaceholder.style.display = "grid";
    this.setText("telemetry-health", "Disconnected");
    this.setText("tracking-state", "No face");
    this.setText("alert-title", "Backend offline");
    this.setText("alert-copy", "Start the backend and reconnect to resume live fatigue monitoring.");
    this.setSyncText(message);
  }

  handleEvents(data) {
    if (!this.connected && data.status) {
      this.pushEvent("Telemetry online", "Live driver monitoring feed connected.", "neutral");
    }

    if ((data.blink_count ?? 0) > this.lastBlinkCount) {
      this.pushEvent("Blink detected", `Total blinks: ${data.blink_count}.`, "neutral");
    }

    if ((data.yawn_alerts ?? 0) > this.lastYawnCount) {
      this.pushEvent("Yawn detected", `Yawns detected: ${data.yawn_alerts}. Consider a short break soon.`, "warning");
    }

    if ((data.drowsy_alerts ?? 0) > this.lastDrowsyCount) {
      this.pushEvent("Drowsiness alert", `Critical fatigue event #${data.drowsy_alerts} triggered. Audible warning enabled.`, "critical");
    }

    if (data.status !== this.currentStatus) {
      this.pushEvent("State changed", `Driver state changed to ${data.status}.`, this.eventSeverityForStatus(data.status));
      this.currentStatus = data.status;
    }

    this.lastBlinkCount = data.blink_count ?? this.lastBlinkCount;
    this.lastYawnCount = data.yawn_alerts ?? this.lastYawnCount;
    this.lastDrowsyCount = data.drowsy_alerts ?? this.lastDrowsyCount;
  }

  eventSeverityForStatus(status) {
    if (status === "DROWSY") {
      return "critical";
    }
    if (status === "WARNING" || status === "CALIBRATING") {
      return "warning";
    }
    return "neutral";
  }

  pushEvent(title, body, severity) {
    const log = document.getElementById("event-log");
    if (!log) {
      return;
    }

    const item = document.createElement("div");
    item.className = `event-item ${severity}`;
    item.innerHTML = `<strong>${title}</strong><span>${body}</span>`;
    log.prepend(item);

    while (log.children.length > 8) {
      log.removeChild(log.lastElementChild);
    }
  }

  updateAlertPresentation(data) {
    const banner = document.getElementById("alert-banner");
    if (!banner) {
      return;
    }

    if (data.status === "DROWSY") {
      banner.className = "alert-banner critical";
      this.setText("alert-title", "Critical drowsiness detected");
      this.setText("alert-copy", "Eyes stayed closed for too long. Alarm is active. Pull over or take a rest break now.");
      return;
    }

    if (data.status === "WARNING") {
      banner.className = "alert-banner warning";
      this.setText("alert-title", "Fatigue warning");
      this.setText("alert-copy", "Blink closure and fatigue signals are rising. Reset posture and prepare for a short stop.");
      return;
    }

    banner.className = "alert-banner calm";
    this.setText("alert-title", data.face_detected ? "Driver monitoring active" : "Searching for face");
    this.setText(
      "alert-copy",
      data.face_detected
        ? "Face tracking is locked and biometric monitoring is running."
        : "Center your face inside the camera frame for better blink and yawn tracking."
    );
  }

  updateCoach(data) {
    if (data.status === "DROWSY") {
      this.setText("coach-title", "Take a break immediately");
      this.setText("coach-copy", "Open-source driver monitoring systems commonly escalate to a loud alarm and urgent rest advice when prolonged eye closure is detected.");
      return;
    }

    if (data.status === "WARNING" || (data.yawn_alerts ?? 0) > 0) {
      this.setText("coach-title", "Fatigue building up");
      this.setText("coach-copy", "Yawns and warning frames suggest reduced alertness. Cool the cabin, stretch your face, and plan a stop within the next few minutes.");
      return;
    }

    this.setText("coach-title", "Monitoring for fatigue");
    this.setText("coach-copy", "The system is watching for prolonged eye closure, repeated yawns, and loss of face tracking.");
  }

  async unlockAudio() {
    if (this.audioUnlocked) {
      return;
    }

    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }
      this.audioUnlocked = true;
    } catch (error) {
      this.audioUnlocked = false;
    }
  }

  updateAudioState(status) {
    if (!this.audioUnlocked) {
      return;
    }

    if (status === "DROWSY") {
      this.startCriticalLoop();
      return;
    }

    this.stopAlertLoops();
  }

  startCriticalLoop() {
    if (this.criticalInterval) {
      return;
    }

    this.stopAlertLoops();
    this.playAlertTone("critical");
    this.criticalInterval = window.setInterval(() => this.playAlertTone("critical"), 1100);
  }

  stopAlertLoops() {
    if (this.criticalInterval) {
      window.clearInterval(this.criticalInterval);
      this.criticalInterval = null;
    }
  }

  playAlertTone(level) {
    if (!this.audioContext) {
      return;
    }

    const now = this.audioContext.currentTime;
    const oscillator = this.audioContext.createOscillator();
    const gainNode = this.audioContext.createGain();

    oscillator.type = "sawtooth";
    oscillator.frequency.setValueAtTime(level === "critical" ? 880 : 660, now);
    oscillator.frequency.linearRampToValueAtTime(level === "critical" ? 660 : 520, now + 0.25);

    gainNode.gain.setValueAtTime(0.0001, now);
    gainNode.gain.linearRampToValueAtTime(level === "critical" ? 0.18 : 0.1, now + 0.03);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + (level === "critical" ? 0.5 : 0.35));

    oscillator.connect(gainNode);
    gainNode.connect(this.audioContext.destination);
    oscillator.start(now);
    oscillator.stop(now + (level === "critical" ? 0.55 : 0.38));
  }

  setSyncText(message) {
    this.setText("sync-text", message);
  }

  setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  }

  formatNumber(value, digits) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
      return digits === 0 ? "0" : (0).toFixed(digits);
    }
    return numericValue.toFixed(digits);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new DriverAttentionDashboard();
});
