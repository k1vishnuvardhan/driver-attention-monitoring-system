# Driver Attention Monitoring

Real-time driver drowsiness monitoring using Python, OpenCV, and MediaPipe.

This project:
- captures live video from a laptop webcam or external USB camera
- tracks the face on the video feed
- calculates blink activity using EAR
- calculates yawn activity using MAR
- shows the processed feed inside the website
- triggers an alert sound when drowsiness is detected

---

## Authors

[![Soumya](https://img.shields.io/badge/GitHub-Soumya-black?logo=github)](https://github.com/soumyapanthangi28)

[![Parvathi](https://img.shields.io/badge/GitHub-Parvathi-black?logo=github)](https://github.com/ParvathiBogati)

[![Vishnu](https://img.shields.io/badge/GitHub-Vishnu-black?logo=github)](https://github.com/k1vishnuvardhan)

---

## Features

- Live face tracking overlay
- Blink counting
- Yawn detection
- Drowsiness state detection
- Browser alert sound when status becomes `DROWSY`
- External camera / dashcam-style camera switching
- Web dashboard with live telemetry

---

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or Python 3.11
- Webcam or external USB camera
- Internet connection for first-time dependency install

Download Python:

https://www.python.org/downloads/release/python-31011/

IMPORTANT:

During installation enable:

[✓] Add Python to PATH

---

## Folder Structure

```text
DriverAttentionMonitoring/
├── assets/
│   └── alarm.wav
├── backend/
├── data/
├── models/
│   └── face_landmarker.task
├── notebooks/
├── src/
│   ├── alert.py
│   ├── dashboard.py
│   ├── detector.py
│   ├── landmark.py
│   ├── landmarks.py
│   └── main.py
├── ui/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── requirements.txt
├── run_web_app.bat
└── README.md
```

---

## Install

Open PowerShell in the project folder and run:

```powershell
winget install Python.Python.3.10

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` does not work on your machine, restart PowerShell and use:

```powershell
py -m venv .venv
.venv\Scripts\activate

py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Recommended:
- Python 3.10
- Python 3.11

Avoid:
- Python 3.13
- Python 3.14
---

## Run

### Easiest way

Double-click:

```text
run_web_app.bat
```

This will:
- start the Python backend
- start the local website server
- open the dashboard in your browser

### Manual way

Open terminal 1:

```powershell
cd C:\path\to\DriverAttentionMonitoring
.venv\Scripts\activate
python src\main.py --headless
```

Open terminal 2:

```powershell
cd C:\path\to\DriverAttentionMonitoring
.venv\Scripts\activate
python -m http.server 5501
```

Open the dashboard:

```text
http://127.0.0.1:5501/ui/index.html
```

Then:
1. Press `Ctrl + F5`
2. Click `Connect Feed`
3. Ensure your face is visible in the frame

---

## External Camera / Dashcam Use

1. Connect the external camera
2. Start the app
3. Open the dashboard
4. Click `Refresh Cameras`
5. Select the desired camera
6. Click `Switch Camera`

Notes:
- `Camera 0` is usually the built-in webcam
- external cameras may appear as `Camera 1`, `Camera 2`, etc.

---

## How To Test

### Backend telemetry

Open:

```text
http://127.0.0.1:5000/telemetry
```

Expected:
- status
- blink_count
- yawn_alerts
- drowsy_alerts
- fps
- camera_name

### Video stream

Open:

```text
http://127.0.0.1:5000/video_feed
```

Expected:
- processed video feed with tracking overlays

### Website

Open:

```text
http://127.0.0.1:5501/ui/index.html
```

Expected:
- live processed feed
- face tracking updates
- blink count updates
- yawn detection updates
- drowsiness alerts
- camera source information
- alert sound when drowsiness is detected

---

## Controls

### Dashboard buttons

- `Connect Feed`
- `Recalibrate`
- `Refresh Cameras`
- `Switch Camera`

### Keyboard controls

When not running in headless mode:

- `Q` = quit
- `R` = recalibrate
- `O` = save open-eye sample
- `C` = save closed-eye sample

---

## Troubleshooting

### Backend Offline

If the website says `Backend Offline`:

1. Ensure backend is running
2. Open:

```text
http://127.0.0.1:5000/telemetry
```

If this does not load, the backend failed to start.

---

### Video stream unavailable

1. Confirm backend is running
2. Hard refresh browser using `Ctrl + F5`
3. Open:

```text
http://127.0.0.1:5000/video_feed
```

---

### No alert sound

Browsers require user interaction before audio playback.

Do:
1. Click anywhere on the page
2. Click `Connect Feed`

---

### Camera not opening

Close apps already using the camera:
- Camera app
- Zoom
- Teams
- Browser tabs using webcam

---

### External camera not shown

Try:
1. unplug camera
2. reconnect camera
3. click `Refresh Cameras`
4. try another USB port

---

### Dependency install issues

Re-activate the environment and run:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Notes

- This project is intended for local real-camera use.
- The browser dashboard reads the processed stream from the Python backend.
- The browser does not directly access the camera in this version.
- Best compatibility is with Python 3.10 or Python 3.11.
