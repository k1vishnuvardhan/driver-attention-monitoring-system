from __future__ import annotations

from dataclasses import dataclass
import time

import cv2


GREEN = (60, 179, 113)
AMBER = (0, 191, 255)
RED = (0, 0, 255)
SLATE = (90, 90, 90)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@dataclass
class DashboardSnapshot:
    status: str
    score_value: float
    threshold_value: float
    mar_value: float
    mar_threshold: float
    yawn_frames: int
    yawn_alerts: int
    closed_frames: int
    blink_count: int
    drowsy_alerts: int
    fps: float
    started_at: float
    integration_mode: str
    calibration_progress: int
    calibration_target: int
    saved_open: int
    saved_closed: int
    capture_label: str


def _panel_color(status: str) -> tuple[int, int, int]:
    if status == "AWAKE":
        return GREEN
    if status == "WARNING":
        return AMBER
    if status == "DROWSY":
        return RED
    return SLATE


def render_dashboard(frame, snapshot: DashboardSnapshot) -> None:
    panel_color = _panel_color(snapshot.status)

    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (440, 365), panel_color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.rectangle(frame, (15, 15), (440, 365), panel_color, 2)

    session_seconds = int(time.time() - snapshot.started_at)
    capture_text = snapshot.capture_label if snapshot.capture_label else "-"

    info_lines = [
        f"Status: {snapshot.status}",
        f"Score: {snapshot.score_value:.3f}",
        f"Threshold: {snapshot.threshold_value:.3f}",
        f"MAR: {snapshot.mar_value:.3f}",
        f"MAR Threshold: {snapshot.mar_threshold:.3f}",
        f"Yawn Frames: {snapshot.yawn_frames}",
        f"Yawn Alerts: {snapshot.yawn_alerts}",
        f"Closed Frames: {snapshot.closed_frames}",
        f"Blinks: {snapshot.blink_count}",
        f"Alerts: {snapshot.drowsy_alerts}",
        f"FPS: {snapshot.fps:.1f}",
        f"Session: {session_seconds}s",
        f"Mode: {snapshot.integration_mode}",
        f"Calibration: {snapshot.calibration_progress}/{snapshot.calibration_target}",
        f"Saved Open: {snapshot.saved_open}",
        f"Saved Closed: {snapshot.saved_closed}",
        f"Last Capture: {capture_text}",
    ]

    for index, line in enumerate(info_lines):
        y = 42 + index * 18
        cv2.putText(frame, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)

    help_text = "Q quit | R recalibrate | O save open | C save closed"
    cv2.putText(frame, help_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
