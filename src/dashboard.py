from __future__ import annotations
import sys
import os
import time
from dataclasses import dataclass
from pathlib import Path
import cv2

# Telemetry HUD panel colors
GREEN = (113, 179, 60)   # BGR Medium Green
AMBER = (10, 159, 255)   # BGR Orange
RED = (48, 59, 255)      # BGR Red
SLATE = (90, 90, 90)     # BGR Grey
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
    pose_label: str
    face_detected: bool
    camera_index: int
    camera_name: str

def _panel_color(status: str) -> tuple[int, int, int]:
    if status in ("AWAKE", "NORMAL"):
        return GREEN
    if status in ("WARNING", "DISTRACTED"):
        return AMBER
    if status in ("DROWSY", "SLEEPING"):
        return RED
    return SLATE

def render_dashboard(frame, snapshot: DashboardSnapshot) -> None:
    """Renders a premium cyberpunk telemetry cockpit HUD overlay on the BGR camera frame."""
    panel_color = _panel_color(snapshot.status)
    h, w, _ = frame.shape

    # Draw semi-transparent glass panel background on the left side
    overlay = frame.copy()
    panel_w = 440
    panel_h = 370
    cv2.rectangle(overlay, (15, 15), (panel_w, panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    
    # Draw panel borders and glowing colored bar representing driver state
    cv2.rectangle(frame, (15, 15), (panel_w, panel_h), (40, 40, 40), 1)
    cv2.rectangle(frame, (15, 15), (panel_w, 22), panel_color, -1)

    session_seconds = int(time.time() - snapshot.started_at)
    capture_text = snapshot.capture_label if snapshot.capture_label else "-"

    info_lines = [
        f"AURA HUD // SYSTEM TELEMETRY GRID",
        f"------------------------------------",
        f"Driver Status    : {snapshot.status}",
        f"Eye EAR Score    : {snapshot.score_value:.4f}",
        f"EAR Threshold    : {snapshot.threshold_value:.4f}",
        f"Mouth MAR Score  : {snapshot.mar_value:.4f}",
        f"MAR Threshold    : {snapshot.mar_threshold:.4f}",
        f"Blink Count      : {snapshot.blink_count}",
        f"Yawn Count       : {snapshot.yawn_alerts} (Frames: {snapshot.yawn_frames})",
        f"Drowsy Alerts    : {snapshot.drowsy_alerts} (Closed: {snapshot.closed_frames})",
        f"Head Pose        : {snapshot.pose_label}",
        f"Face Tracking    : {'LOCKED' if snapshot.face_detected else 'SEARCHING'}",
        f"Camera Source    : {snapshot.camera_name}",
        f"Optical FPS      : {snapshot.fps:.1f} fps",
        f"Session Duration : {session_seconds} seconds",
        f"Biometric Mode   : {snapshot.integration_mode}",
        f"Self Calibration : {snapshot.calibration_progress}/{snapshot.calibration_target} Frames",
        f"Captured Open    : {snapshot.saved_open} frames",
        f"Captured Closed  : {snapshot.saved_closed} frames",
        f"Capture Actions  : {capture_text}",
    ]

    for index, line in enumerate(info_lines):
        y = 48 + index * 18
        if index == 0:
            # Highlight title header text
            cv2.putText(frame, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, panel_color, 1, cv2.LINE_AA)
        elif index == 1:
            cv2.putText(frame, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
        else:
            # Regular metrics text
            cv2.putText(frame, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    help_text = "Q: Exit | R: Recalibrate | O: Capture Open | C: Capture Closed"
    cv2.putText(frame, help_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)


# ====================================================
# STREAMLIT COCKPIT DASHBOARD APPLICATION
# ====================================================
# Checks if executing under Streamlit directly
if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT") is not None or __name__ == "__main__":
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        
        # Setup page configuration
        st.set_page_config(
            page_title="AURA // Driver Attention Cockpit UI",
            page_icon="👁️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0c0f12;
                color: #e3e8ed;
            }
            .metric-box {
                background-color: #161b22;
                border: 1px solid #30363d;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("👁️ AURA Driver Attention Monitor")
        st.subheader("Automotive HMI Telemetry Stream & Live Analytics Cockpit")

        # Sidebar navigation and trip configurations
        st.sidebar.title("AURA Controls")
        paired_vehicle = st.sidebar.selectbox("Paired Vehicle", ["Tesla Model S Plaid (CAN)", "Rivian R1S Dual-Motor", "MBEQE HyperScreen", "Standalone Cockpit HUD"])
        st.sidebar.info(f"System actively linked to: {paired_vehicle}")

        # Streamlit metrics layout
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Attention Level", value="98.5%", delta="Normal State")
        with col2:
            st.metric(label="Trip Blinks", value="42 blinks", delta="Stable frequency")
        with col3:
            st.metric(label="Yawn count", value="2 yawns", delta="AC cool boost idle")
        with col4:
            st.metric(label="Active Warnings", value="0 active", delta="No incidents")

        st.markdown("### Telemetry Signal Stream")
        
        # Generate simulation chart of eye aspect ratio
        chart_data = pd.DataFrame(
            np.random.randn(50, 2) * 0.02 + [0.31, 0.12],
            columns=["Eye Aspect Ratio (EAR)", "Mouth Aspect Ratio (MAR)"]
        )
        st.line_chart(chart_data)

        st.markdown("### Trip Incident Logs")
        incidents = pd.DataFrame([
            {"Timestamp": "16:21:05", "Event": "Gaze off-road deflection", "Duration": "1.5s", "Severity": "WARNING", "Action Taken": "Acoustic Chime Sent"},
            {"Timestamp": "16:15:32", "Event": "Excessive Yawning", "Duration": "2.0s", "Severity": "WARNING", "Action Taken": "Cabin Air Speed Boosted"},
            {"Timestamp": "16:02:18", "Event": "Normal Eye Calibration Complete", "Duration": "3.0s", "Severity": "INFO", "Action Taken": "Baselines Calibrated"}
        ])
        st.table(incidents)

    except ImportError:
        # Streamlit is not installed, fail silently or show error if run as script
        if __name__ == "__main__":
            print("Error: Streamlit must be installed to launch the Streamlit Web Cockpit.")
            print("Run: pip install streamlit pandas plotly")
