# Driver Attention Monitoring

Real-time driver drowsiness monitoring using Python, OpenCV, and MediaPipe.

This project:
- captures live video from a laptop webcam or external USB camera
- tracks the face on the video feed
- calculates blink activity using EAR
- calculates yawn activity using MAR
- shows the processed feed inside the website
- triggers an alert sound when drowsiness is detected

## Authors

[![Soumya](https://img.shields.io/badge/GitHub-Soumya-black?logo=github)](https://github.com/soumyapanthangi28)

[![Parvathi](https://img.shields.io/badge/GitHub-Parvathi-black?logo=github)](https://github.com/ParvathiBogati)

[![Vishnu](https://img.shields.io/badge/GitHub-Vishnu-black?logo=github)](https://github.com/k1vishnuvardhan)

## Features

- Live face tracking overlay
- Blink counting
- Yawn detection
- Drowsiness state detection
- Browser alert sound when status becomes `DROWSY`
- External camera / dashcam-style camera switching
- Web dashboard with live telemetry

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or Python 3.11
- A webcam or supported external USB camera
- Internet connection for the first dependency install

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

## Install

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `python` does not work on your machine, use:

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

That is the main dependency install command:

```powershell
pip install -r requirements.txt
```

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
1. Press `Ctrl + F5` once
2. Click `Connect Feed`
3. Make sure your face is visible in the frame

## External Camera / Dashcam Use

If you want to use an external USB webcam or a dashcam capture device:

1. Connect the external camera to the computer
2. Start the app
3. Open the dashboard
4. Click `Refresh Cameras`
5. Select the desired camera source
6. Click `Switch Camera`

Notes:
- `Camera 0` is usually the built-in webcam
- external cameras often appear as `Camera 1`, `Camera 2`, and so on
- the device must be visible to Windows as a camera or capture device

## How To Test

### Test 1: Backend telemetry

Open:

```text
http://127.0.0.1:5000/telemetry
```

You should see JSON output with values like:
- `status`
- `blink_count`
- `yawn_alerts`
- `drowsy_alerts`
- `fps`
- `camera_name`

### Test 2: Video stream

Open:

```text
http://127.0.0.1:5000/video_feed
```

You should see the processed video feed with tracking overlays.

### Test 3: Website

Open:

```text
http://127.0.0.1:5501/ui/index.html
```

Expected result:
- live processed feed appears in the page
- face tracking status updates
- blink count updates
- yawn count updates
- drowsiness alerts update
- camera source is shown
- alert sound plays when `DROWSY` state is reached

## Controls

### Dashboard buttons

- `Connect Feed`:
  connect the website to the backend
- `Recalibrate`:
  reset the adaptive eye baseline
- `Refresh Cameras`:
  rescan available camera sources
- `Switch Camera`:
  switch to the selected camera source

### Keyboard controls in backend window

When not running in headless mode:

- `Q` = quit
- `R` = recalibrate
- `O` = save open-eye sample
- `C` = save closed-eye sample

## Troubleshooting

### Backend Offline

If the website says `Backend Offline`:

1. Make sure `python src\main.py --headless` is still running
2. Open:

```text
http://127.0.0.1:5000/telemetry
```

If this page does not load, the backend is not running correctly.

### Video stream unavailable

If the site says `Video stream unavailable`:

1. Confirm the backend is running
2. Hard refresh the browser with `Ctrl + F5`
3. Open:

```text
http://127.0.0.1:5000/video_feed
```

If that also fails, the backend is not producing frames.

### No alert sound

Browsers usually require one normal user interaction before audio can play.

Do this:
1. Click anywhere on the page
2. Click `Connect Feed`

After that, when the system enters `DROWSY`, the alert sound should play automatically.

### Camera not opening

Close other apps that may already be using the camera:
- Camera app
- Zoom
- Teams
- browser tabs using camera

### External camera not shown

Try:
1. unplug the camera
2. plug it in again
3. click `Refresh Cameras`
4. try another USB port

### Dependency install issues

Re-activate the virtual environment and run:

```powershell
pip install -r requirements.txt
```

If needed:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Notes

- This project is designed for local real-camera use.
- The browser dashboard reads the processed stream from the Python backend.
- The browser does not directly access the camera in this version.
