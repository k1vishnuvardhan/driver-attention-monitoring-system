from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, TypeGuard
import inspect
import math
import time
import winsound
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np

from dashboard import DashboardSnapshot, render_dashboard


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 308, 14, 311]

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
MIN_BLINK_FRAMES = 2
WARNING_FRAMES = 15
DROWSY_FRAMES = 30
CALIBRATION_FRAMES = 45
YAWN_FRAMES = 12
MAR_THRESHOLD = 0.60

GREEN = (60, 179, 113)
CYAN = (255, 220, 80)
AMBER = (0, 191, 255)
RED = (60, 70, 255)
ALARM_PATH = Path(__file__).resolve().parent.parent / "assets" / "alarm.wav"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OPEN_DATA_DIR = DATA_ROOT / "open"
CLOSED_DATA_DIR = DATA_ROOT / "closed"
FACE_LANDMARKER_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class TelemetryState:
    def __init__(self):
        self.latest_frame = None
        self.latest_snapshot = None
        self.lock = threading.Lock()

global_telemetry = TelemetryState()


global_recalibrate_trigger = False
global_save_open_trigger = False
global_save_closed_trigger = False
global_camera_switch_request: int | None = None


@dataclass
class CameraSource:
    index: int
    label: str


def create_video_capture(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def list_available_cameras(max_cameras: int = 6) -> list[CameraSource]:
    sources: list[CameraSource] = []
    for index in range(max_cameras):
        cap = create_video_capture(index)
        if cap.isOpened():
            sources.append(CameraSource(index=index, label=f"Camera {index}"))
        cap.release()
    return sources


def open_best_available_camera(preferred_index: int = 0) -> tuple[cv2.VideoCapture, int]:
    preferred_capture = create_video_capture(preferred_index)
    if preferred_capture.isOpened():
        return preferred_capture, preferred_index

    preferred_capture.release()
    for source in list_available_cameras():
        if source.index == preferred_index:
            continue
        fallback_capture = create_video_capture(source.index)
        if fallback_capture.isOpened():
            return fallback_capture, source.index
        fallback_capture.release()

    return preferred_capture, preferred_index


class TelemetryHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Mute logging output in console
        pass

    def do_GET(self):
        global global_recalibrate_trigger, global_save_open_trigger, global_save_closed_trigger, global_camera_switch_request
        parsed_url = urlparse(self.path)
        request_path = parsed_url.path
        query_items = dict(item.split("=", 1) for item in parsed_url.query.split("&") if "=" in item)
        
        if request_path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            while True:
                with global_telemetry.lock:
                    frame_bytes = global_telemetry.latest_frame
                
                if frame_bytes:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode('utf-8'))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                    except Exception:
                        break
                
                # Stream frames at ~30 FPS
                time.sleep(0.033)
                
        elif request_path == '/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            with global_telemetry.lock:
                snap = global_telemetry.latest_snapshot
            
            if snap:
                data = {
                    "status": snap.status,
                    "score_value": snap.score_value,
                    "threshold_value": snap.threshold_value,
                    "mar_value": snap.mar_value,
                    "mar_threshold": snap.mar_threshold,
                    "yawn_frames": snap.yawn_frames,
                    "yawn_alerts": snap.yawn_alerts,
                    "closed_frames": snap.closed_frames,
                    "blink_count": snap.blink_count,
                    "drowsy_alerts": snap.drowsy_alerts,
                    "fps": snap.fps,
                    "integration_mode": snap.integration_mode,
                    "pose_label": snap.pose_label,
                    "face_detected": snap.face_detected,
                    "camera_index": getattr(snap, "camera_index", 0),
                    "camera_name": getattr(snap, "camera_name", "Camera 0"),
                }
                self.wfile.write(json.dumps(data).encode('utf-8'))
            else:
                self.wfile.write(b'{}')

        elif request_path == '/cameras':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            cameras = [{"index": source.index, "label": source.label} for source in list_available_cameras()]
            self.wfile.write(json.dumps({"cameras": cameras}).encode('utf-8'))
                
        elif request_path == '/recalibrate':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            global_recalibrate_trigger = True
            self.wfile.write(b'{"status":"recalibrated"}')
            
        elif request_path == '/save_open':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            global_save_open_trigger = True
            self.wfile.write(b'{"status":"save_open_queued"}')
            
        elif request_path == '/save_closed':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            global_save_closed_trigger = True
            self.wfile.write(b'{"status":"save_closed_queued"}')

        elif request_path == '/select_camera':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                camera_index = int(query_items.get("index", "0"))
                global_camera_switch_request = camera_index
                self.wfile.write(json.dumps({"status": "camera_switch_queued", "camera_index": camera_index}).encode('utf-8'))
            except ValueError:
                self.wfile.write(b'{"status":"invalid_camera_index"}')
            
        else:
            self.send_response(404)
            self.end_headers()


@dataclass
class DetectorState:
    blink_count: int = 0
    drowsy_alerts: int = 0
    closed_frames: int = 0
    blink_frames: int = 0
    alarm_on: bool = False
    status: str = "CALIBRATING"
    started_at: float = field(default_factory=time.time)
    open_reference: float = 0.0
    closed_reference: float = 0.0
    threshold_value: float = 0.20
    current_score: float = 0.0
    calibration_frames: int = 0
    saved_open: int = 0
    saved_closed: int = 0
    last_capture_label: str = ""
    last_capture_at: float = 0.0
    mar_value: float = 0.0
    yawn_frames: int = 0
    yawn_alerts: int = 0
    pose_label: str = "CENTER"
    face_detected: bool = False
    camera_index: int = 0
    camera_name: str = "Camera 0"


@dataclass
class IntegrationContext:
    landmark_module: ModuleType | None = None
    detector_module: ModuleType | None = None
    mode: str = "internal"


@dataclass
class LandmarkBackend:
    name: str
    runner: Any
    uses_mediapipe_results: bool = True


@dataclass
class OpenCvLandmarkResult:
    left_eye: list[tuple[int, int]] | None = None
    right_eye: list[tuple[int, int]] | None = None
    left_eye_crop: np.ndarray[Any, Any] | None = None
    right_eye_crop: np.ndarray[Any, Any] | None = None
    score: float = 0.0
    mouth_points: list[tuple[int, int]] | None = None
    face_box: tuple[int, int, int, int] | None = None


@dataclass
class FaceMeshResult:
    multi_face_landmarks: list[Any] | None = None


def ensure_data_dirs() -> None:
    OPEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLOSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "sessions").mkdir(parents=True, exist_ok=True)


def euclidean_distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return math.dist(point_a, point_b)


def to_pixel(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def to_uniform(landmark) -> tuple[float, float]:
    return float(landmark.x * 1000.0), float(landmark.y * 1000.0)


def calculate_ear(eye_points: list[tuple[int, int]]) -> float:
    vertical_1 = euclidean_distance(eye_points[1], eye_points[5])
    vertical_2 = euclidean_distance(eye_points[2], eye_points[4])
    horizontal = euclidean_distance(eye_points[0], eye_points[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(mouth_points: list[tuple[int, int]]) -> float:
    vertical_1 = euclidean_distance(mouth_points[1], mouth_points[5])
    vertical_2 = euclidean_distance(mouth_points[2], mouth_points[4])
    horizontal = euclidean_distance(mouth_points[0], mouth_points[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def eye_points_from_box(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    return [
        (x, y + height // 2),
        (x + width // 4, y + height // 4),
        (x + (3 * width) // 4, y + height // 4),
        (x + width, y + height // 2),
        (x + (3 * width) // 4, y + (3 * height) // 4),
        (x + width // 4, y + (3 * height) // 4),
    ]


def get_haar_cascade_path(filename: str) -> str:
    cv2_data = getattr(cv2, "data", None)
    if cv2_data is None:
        raise RuntimeError("OpenCV was imported without the cv2.data helper path.")
    return str(Path(cv2_data.haarcascades) / filename)


def extract_eye_points(face_landmarks, frame_width: int, frame_height: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    landmark_items = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    left_eye = [to_pixel(landmark_items[index], frame_width, frame_height) for index in LEFT_EYE]
    right_eye = [to_pixel(landmark_items[index], frame_width, frame_height) for index in RIGHT_EYE]
    return left_eye, right_eye


def extract_mouth_points(face_landmarks, frame_width: int, frame_height: int) -> list[tuple[int, int]]:
    landmark_items = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    return [to_pixel(landmark_items[index], frame_width, frame_height) for index in MOUTH]


def extract_eye_points_uniform(face_landmarks) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    landmark_items = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    left_eye = [to_uniform(landmark_items[index]) for index in LEFT_EYE]
    right_eye = [to_uniform(landmark_items[index]) for index in RIGHT_EYE]
    return left_eye, right_eye


def extract_mouth_points_uniform(face_landmarks) -> list[tuple[float, float]]:
    landmark_items = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    return [to_uniform(landmark_items[index]) for index in MOUTH]



def draw_eye_outline(frame, eye_points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for point in eye_points:
        cv2.circle(frame, point, 2, color, -1)
    contour = np.array(eye_points, dtype=np.int32)
    cv2.polylines(frame, [cv2.convexHull(contour)], True, color, 1)


def draw_mouth_outline(frame, mouth_points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    contour = np.array(mouth_points, dtype=np.int32)
    cv2.polylines(frame, [cv2.convexHull(contour)], True, color, 1)
    for point in mouth_points:
        cv2.circle(frame, point, 2, color, -1)


def draw_face_box(frame, face_box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, width, height = face_box
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)


def draw_face_mesh(frame, face_landmarks) -> None:
    face_mesh_api = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    if face_mesh_api is None:
        return

    landmark_items = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    frame_height, frame_width = frame.shape[:2]
    for start_index, end_index in face_mesh_api.FACEMESH_TESSELATION:
        start_point = to_pixel(landmark_items[start_index], frame_width, frame_height)
        end_point = to_pixel(landmark_items[end_index], frame_width, frame_height)
        cv2.line(frame, start_point, end_point, CYAN, 1, cv2.LINE_AA)


def compute_face_box(face_landmarks, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    landmark_items = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    x_values = [int(landmark.x * frame_width) for landmark in landmark_items]
    y_values = [int(landmark.y * frame_height) for landmark in landmark_items]
    min_x = max(min(x_values) - 10, 0)
    min_y = max(min(y_values) - 10, 0)
    max_x = min(max(x_values) + 10, frame_width - 1)
    max_y = min(max(y_values) + 10, frame_height - 1)
    return min_x, min_y, max(max_x - min_x, 1), max(max_y - min_y, 1)


def estimate_pose_label(face_landmarks) -> str:
    landmark_items = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    nose = landmark_items[4]
    left = landmark_items[234]
    right = landmark_items[454]
    top = landmark_items[10]
    bottom = landmark_items[152]

    horizontal_span = max(right.x - left.x, 1e-6)
    vertical_span = max(bottom.y - top.y, 1e-6)
    horizontal_ratio = (nose.x - left.x) / horizontal_span
    vertical_ratio = (nose.y - top.y) / vertical_span

    if vertical_ratio > 0.60:
        return "LOOKING DOWN"
    if horizontal_ratio < 0.38:
        return "LOOKING RIGHT"
    if horizontal_ratio > 0.62:
        return "LOOKING LEFT"
    return "CENTER"


def annotate_tracking_label(frame, label: str, face_box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, _, _ = face_box
    text_y = max(y - 12, 20)
    cv2.putText(frame, f"TRACKING: {label}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def start_alarm() -> None:
    if ALARM_PATH.exists():
        winsound.PlaySound(str(ALARM_PATH), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
    else:
        winsound.Beep(2200, 400)


def stop_alarm() -> None:
    winsound.PlaySound(None, 0)


def update_threshold(state: DetectorState, score_value: float) -> None:
    state.current_score = score_value

    if state.calibration_frames < CALIBRATION_FRAMES:
        state.open_reference = max(state.open_reference, score_value)
        if state.closed_reference == 0.0:
            state.closed_reference = score_value
        else:
            state.closed_reference = min(state.closed_reference, score_value)
        state.calibration_frames += 1
        state.threshold_value = max(state.open_reference * 0.72, 0.10)
        state.status = "CALIBRATING"
        return

    state.open_reference = max(state.open_reference * 0.98, score_value)
    state.threshold_value = max(state.open_reference * 0.72, 0.10)


def update_state(state: DetectorState, score_value: float) -> None:
    update_threshold(state, score_value)
    if state.calibration_frames < CALIBRATION_FRAMES:
        state.closed_frames = 0
        state.blink_frames = 0
        if state.alarm_on:
            stop_alarm()
            state.alarm_on = False
        return

    if score_value < state.threshold_value:
        state.closed_frames += 1
        state.blink_frames += 1
    else:
        if MIN_BLINK_FRAMES <= state.blink_frames < WARNING_FRAMES:
            state.blink_count += 1
        state.closed_frames = 0
        state.blink_frames = 0

    if state.closed_frames >= DROWSY_FRAMES:
        if state.status != "DROWSY":
            state.drowsy_alerts += 1
        state.status = "DROWSY"
        if not state.alarm_on:
            start_alarm()
            state.alarm_on = True
    elif state.closed_frames >= WARNING_FRAMES:
        state.status = "WARNING"
        if state.alarm_on:
            stop_alarm()
            state.alarm_on = False
    else:
        state.status = "AWAKE"
        if state.alarm_on:
            stop_alarm()
            state.alarm_on = False


def update_mar_state(state: DetectorState, mar_value: float) -> None:
    state.mar_value = mar_value
    if mar_value > MAR_THRESHOLD:
        state.yawn_frames += 1
        if state.yawn_frames == YAWN_FRAMES:
            state.yawn_alerts += 1
    else:
        state.yawn_frames = 0


def load_optional_module(*names: str) -> ModuleType | None:
    for name in names:
        try:
            return import_module(name)
        except ImportError:
            continue
    return None


def call_with_supported_args(func, **kwargs):
    signature = inspect.signature(func)
    supported_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(**supported_kwargs)


def load_integration_context() -> IntegrationContext:
    use_external_modules = False
    if not use_external_modules:
        return IntegrationContext(mode="internal")

    landmark_module = load_optional_module("landmarks", "src.landmarks")
    detector_module = load_optional_module("detector", "src.detector")

    parts = []
    if landmark_module:
        parts.append("landmarks")
    if detector_module:
        parts.append("detector")

    mode = " + ".join(parts) if parts else "internal"
    return IntegrationContext(landmark_module=landmark_module, detector_module=detector_module, mode=mode)


def create_external_landmark_backend(module: ModuleType) -> LandmarkBackend | None:
    candidate_names = [
        "create_landmark_backend",
        "create_face_mesh",
        "create_face_landmarker",
        "build_landmark_backend",
    ]

    for name in candidate_names:
        factory = getattr(module, name, None)
        if not callable(factory):
            continue

        backend = call_with_supported_args(factory)
        if backend is not None:
            return LandmarkBackend(name=f"external:{name}", runner=backend, uses_mediapipe_results=False)

    return None


def create_internal_landmark_backend() -> LandmarkBackend:
    if FACE_LANDMARKER_MODEL_PATH.exists():
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        runner = vision.FaceLandmarker.create_from_options(options)
        return LandmarkBackend(name="mediapipe.tasks.face_landmarker", runner=runner, uses_mediapipe_results=True)

    face_mesh_api = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    if face_mesh_api is None:
        face_cascade = cv2.CascadeClassifier(get_haar_cascade_path("haarcascade_frontalface_default.xml"))
        eye_cascade = cv2.CascadeClassifier(get_haar_cascade_path("haarcascade_eye_tree_eyeglasses.xml"))
        if face_cascade.empty() or eye_cascade.empty():
            raise RuntimeError(
                "Could not load an internal landmark backend. MediaPipe Face Landmarker, MediaPipe Face Mesh, "
                "and the OpenCV Haar cascade fallback are all unavailable."
            )

        return LandmarkBackend(
            name="opencv.haarcascade",
            runner={"face_cascade": face_cascade, "eye_cascade": eye_cascade},
            uses_mediapipe_results=False,
        )

    runner = face_mesh_api.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return LandmarkBackend(name="mediapipe.solutions.face_mesh", runner=runner, uses_mediapipe_results=True)


def create_landmark_backend(integration: IntegrationContext) -> LandmarkBackend:
    if integration.landmark_module:
        external_backend = create_external_landmark_backend(integration.landmark_module)
        if external_backend is not None:
            integration.mode = f"{integration.mode} | backend"
            return external_backend

    return create_internal_landmark_backend()


def close_landmark_backend(backend: LandmarkBackend) -> None:
    close_method = getattr(backend.runner, "close", None)
    if callable(close_method):
        close_method()


def process_landmarks(backend: LandmarkBackend, frame_rgb) -> FaceMeshResult | OpenCvLandmarkResult:
    if backend.name == "mediapipe.tasks.face_landmarker":
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if not hasattr(backend, "last_timestamp"):
            backend.last_timestamp = 0
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= backend.last_timestamp:
            timestamp_ms = backend.last_timestamp + 1
        backend.last_timestamp = timestamp_ms
        raw_results = backend.runner.detect_for_video(mp_image, timestamp_ms)
        return FaceMeshResult(multi_face_landmarks=getattr(raw_results, "face_landmarks", None))

    if backend.uses_mediapipe_results:
        raw_results = backend.runner.process(frame_rgb)
        return FaceMeshResult(multi_face_landmarks=getattr(raw_results, "multi_face_landmarks", None))

    if backend.name == "opencv.haarcascade":
        return process_landmarks_with_haar(backend.runner, frame_rgb)

    runner = backend.runner
    if callable(runner):
        try:
            return coerce_landmark_result(call_with_supported_args(runner, frame_rgb=frame_rgb))
        except ValueError:
            return coerce_landmark_result(runner(frame_rgb))

    process_method = getattr(runner, "process", None)
    if callable(process_method):
        return coerce_landmark_result(call_with_supported_args(process_method, frame_rgb=frame_rgb))

    detect_method = getattr(runner, "detect", None)
    if callable(detect_method):
        return coerce_landmark_result(call_with_supported_args(detect_method, frame_rgb=frame_rgb))

    raise RuntimeError("Unsupported landmark backend. Expected a callable, .process(), or .detect().")


def crop_box(image: np.ndarray[Any, Any], x: int, y: int, width: int, height: int) -> np.ndarray[Any, Any]:
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + width, image.shape[1])
    y1 = min(y + height, image.shape[0])
    return image[y0:y1, x0:x1]


def estimate_eye_openness(eye_crop: np.ndarray[Any, Any] | None) -> float:
    if eye_crop is None or eye_crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = float(np.count_nonzero(threshold)) / float(threshold.size)
    aspect_ratio = eye_crop.shape[0] / max(eye_crop.shape[1], 1)
    return (dark_ratio * 0.7) + (aspect_ratio * 0.3)


def process_landmarks_with_haar(runner: dict[str, Any], frame_rgb) -> OpenCvLandmarkResult:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = runner["face_cascade"].detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(120, 120))
    if len(faces) == 0:
        return OpenCvLandmarkResult()

    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    face_roi_gray = gray[y : y + height, x : x + width]
    eyes = runner["eye_cascade"].detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 20))

    eye_boxes = []
    for ex, ey, ew, eh in eyes:
        center_x = x + ex + ew / 2
        center_y = y + ey + eh / 2
        if center_y < y + (height * 0.65):
            eye_boxes.append((x + ex, y + ey, ew, eh, center_x))

    if len(eye_boxes) < 2:
        return OpenCvLandmarkResult()

    eye_boxes.sort(key=lambda item: item[4])
    left_box, right_box = eye_boxes[:2]

    left_eye = eye_points_from_box(left_box[0], left_box[1], left_box[2], left_box[3])
    right_eye = eye_points_from_box(right_box[0], right_box[1], right_box[2], right_box[3])
    left_crop = crop_box(frame_bgr, left_box[0], left_box[1], left_box[2], left_box[3])
    right_crop = crop_box(frame_bgr, right_box[0], right_box[1], right_box[2], right_box[3])
    score = (estimate_eye_openness(left_crop) + estimate_eye_openness(right_crop)) / 2.0

    mouth_width = max(width // 2, 1)
    mouth_height = max(height // 6, 1)
    mouth_x = x + (width - mouth_width) // 2
    mouth_y = y + int(height * 0.68)
    mouth_points = eye_points_from_box(mouth_x, mouth_y, mouth_width, mouth_height)

    return OpenCvLandmarkResult(
        left_eye=left_eye,
        right_eye=right_eye,
        left_eye_crop=left_crop,
        right_eye_crop=right_crop,
        score=score,
        mouth_points=mouth_points,
        face_box=(x, y, width, height),
    )


def coerce_landmark_result(raw_result: Any) -> FaceMeshResult | OpenCvLandmarkResult:
    if isinstance(raw_result, OpenCvLandmarkResult):
        return raw_result
    if isinstance(raw_result, FaceMeshResult):
        return raw_result
    return FaceMeshResult(multi_face_landmarks=getattr(raw_result, "multi_face_landmarks", None))


def has_face_landmarks(results: FaceMeshResult | OpenCvLandmarkResult) -> TypeGuard[FaceMeshResult]:
    return isinstance(results, FaceMeshResult) and results.multi_face_landmarks is not None and len(results.multi_face_landmarks) > 0


def external_eye_points(
    module: ModuleType,
    frame,
    frame_rgb,
    results,
    face_landmarks,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    candidate_names = [
        "extract_eye_points",
        "get_eye_points",
        "get_eye_landmarks",
        "extract_landmarks",
    ]

    for name in candidate_names:
        func = getattr(module, name, None)
        if not callable(func):
            continue

        output = call_with_supported_args(
            func,
            frame=frame,
            frame_rgb=frame_rgb,
            results=results,
            face_landmarks=face_landmarks,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            left_eye_indices=LEFT_EYE,
            right_eye_indices=RIGHT_EYE,
        )

        if isinstance(output, tuple) and len(output) == 2:
            return output

    return None


def external_ear_value(
    module: ModuleType,
    left_eye: list[tuple[int, int]],
    right_eye: list[tuple[int, int]],
) -> float | None:
    candidate_names = [
        "calculate_ear",
        "compute_ear",
        "eye_aspect_ratio",
    ]

    for name in candidate_names:
        func = getattr(module, name, None)
        if not callable(func):
            continue

        output = call_with_supported_args(
            func,
            left_eye=left_eye,
            right_eye=right_eye,
            eye_points=left_eye,
            left_eye_points=left_eye,
            right_eye_points=right_eye,
        )

        if isinstance(output, (int, float)):
            return float(output)
        if isinstance(output, tuple) and output and all(isinstance(item, (int, float)) for item in output[:2]):
            return float(sum(output[:2]) / min(len(output), 2))

    return None


def resolve_eye_points(
    integration: IntegrationContext,
    frame,
    frame_rgb,
    results,
    face_landmarks,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if isinstance(results, OpenCvLandmarkResult) and results.left_eye and results.right_eye:
        return results.left_eye, results.right_eye

    if integration.landmark_module:
        points = external_eye_points(
            integration.landmark_module,
            frame=frame,
            frame_rgb=frame_rgb,
            results=results,
            face_landmarks=face_landmarks,
        )
        if points:
            return points

    return extract_eye_points(face_landmarks, frame.shape[1], frame.shape[0])


def resolve_score_value(
    integration: IntegrationContext,
    results: FaceMeshResult | OpenCvLandmarkResult,
    left_eye: list[tuple[int, int]],
    right_eye: list[tuple[int, int]],
) -> float:
    if isinstance(results, OpenCvLandmarkResult) and results.score > 0.0:
        return results.score

    if integration.detector_module:
        external_value = external_ear_value(integration.detector_module, left_eye, right_eye)
        if external_value is not None:
            return external_value

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    return (left_ear + right_ear) / 2.0


def resolve_mar_value(
    integration: IntegrationContext,
    results: FaceMeshResult | OpenCvLandmarkResult,
    mouth_points: list[tuple[int, int]],
) -> float:
    if integration.detector_module:
        external_calculate_mar = getattr(integration.detector_module, "calculate_mar", None)
        if callable(external_calculate_mar):
            external_value = call_with_supported_args(external_calculate_mar, mouth_points=mouth_points)
            if isinstance(external_value, (int, float)):
                return float(external_value)
    return calculate_mar(mouth_points)


def build_snapshot(state: DetectorState, fps: float, integration: IntegrationContext) -> DashboardSnapshot:
    return DashboardSnapshot(
        status=state.status,
        score_value=state.current_score,
        threshold_value=state.threshold_value,
        mar_value=state.mar_value,
        mar_threshold=MAR_THRESHOLD,
        yawn_frames=state.yawn_frames,
        yawn_alerts=state.yawn_alerts,
        closed_frames=state.closed_frames,
        blink_count=state.blink_count,
        drowsy_alerts=state.drowsy_alerts,
        fps=fps,
        started_at=state.started_at,
        integration_mode=integration.mode,
        calibration_progress=min(state.calibration_frames, CALIBRATION_FRAMES),
        calibration_target=CALIBRATION_FRAMES,
        saved_open=state.saved_open,
        saved_closed=state.saved_closed,
        capture_label=state.last_capture_label,
        pose_label=state.pose_label,
        face_detected=state.face_detected,
        camera_index=state.camera_index,
        camera_name=state.camera_name,
    )


def reset_face_state(state: DetectorState) -> None:
    state.status = "NO FACE"
    state.closed_frames = 0
    state.blink_frames = 0
    state.yawn_frames = 0
    state.pose_label = "SEARCHING"
    state.face_detected = False
    if state.alarm_on:
        stop_alarm()
        state.alarm_on = False


def combine_eye_crops(results: OpenCvLandmarkResult) -> np.ndarray[Any, Any] | None:
    if results.left_eye_crop is None or results.right_eye_crop is None:
        return None

    target_height = max(results.left_eye_crop.shape[0], results.right_eye_crop.shape[0])
    left = cv2.resize(results.left_eye_crop, (64, target_height))
    right = cv2.resize(results.right_eye_crop, (64, target_height))
    return np.hstack([left, right])


def save_eye_sample(results: OpenCvLandmarkResult, label: str, state: DetectorState) -> None:
    combined = combine_eye_crops(results)
    if combined is None or combined.size == 0:
        state.last_capture_label = "capture_failed"
        state.last_capture_at = time.time()
        return

    directory = OPEN_DATA_DIR if label == "open" else CLOSED_DATA_DIR
    timestamp = int(time.time() * 1000)
    filename = directory / f"{label}_{timestamp}.png"
    cv2.imwrite(str(filename), combined)

    if label == "open":
        state.saved_open += 1
    else:
        state.saved_closed += 1

    state.last_capture_label = f"saved_{label}"
    state.last_capture_at = time.time()


def handle_keypress(key_code: int, latest_results: FaceMeshResult | OpenCvLandmarkResult, state: DetectorState) -> bool:
    if key_code == ord("q"):
        return False

    if key_code == ord("r"):
        state.open_reference = 0.0
        state.closed_reference = 0.0
        state.threshold_value = 0.20
        state.calibration_frames = 0
        state.closed_frames = 0
        state.blink_frames = 0
        state.status = "CALIBRATING"
        state.last_capture_label = "recalibrated"
        state.last_capture_at = time.time()
        return True

    if isinstance(latest_results, OpenCvLandmarkResult):
        if key_code == ord("o"):
            save_eye_sample(latest_results, "open", state)
        elif key_code == ord("c"):
            save_eye_sample(latest_results, "closed", state)

    return True


def main() -> None:
    global global_recalibrate_trigger, global_save_open_trigger, global_save_closed_trigger, global_camera_switch_request
    ensure_data_dirs()
    integration = load_integration_context()
    landmark_backend = create_landmark_backend(integration)
    if integration.mode == "internal":
        integration.mode = landmark_backend.name

    current_camera_index = 0
    cap, current_camera_index = open_best_available_camera(current_camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    # Start HTTP Telemetry and Video Feed Server on port 5000 in background
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 5000), TelemetryHTTPHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print("AURA TELEMETRY // Backend HTTP server active on http://127.0.0.1:5000")
    except Exception as e:
        print(f"AURA TELEMETRY // Failed to start backend HTTP server: {e}")

    state = DetectorState()
    state.camera_index = current_camera_index
    state.camera_name = f"Camera {current_camera_index}"
    previous_time = time.time()
    failed_read_count = 0

    try:
        while True:
            if global_camera_switch_request is not None and global_camera_switch_request != current_camera_index:
                requested_camera = global_camera_switch_request
                replacement = create_video_capture(requested_camera)
                if replacement.isOpened():
                    cap.release()
                    cap = replacement
                    current_camera_index = requested_camera
                    state.camera_index = current_camera_index
                    state.camera_name = f"Camera {current_camera_index}"
                    state.last_capture_label = f"camera_{current_camera_index}"
                    state.last_capture_at = time.time()
                    print(f"AURA TELEMETRY // Switched to camera source {current_camera_index}.")
                else:
                    replacement.release()
                    print(f"AURA TELEMETRY // Failed to switch to camera source {requested_camera}.")
                global_camera_switch_request = None

            success, frame = cap.read()
            if not success:
                failed_read_count += 1
                print(f"Failed to read from camera {current_camera_index}. Retry {failed_read_count}/15.")

                if failed_read_count >= 5:
                    available_sources = list_available_cameras()
                    fallback_indices = [source.index for source in available_sources if source.index != current_camera_index]
                    switched = False
                    for fallback_index in fallback_indices:
                        replacement = create_video_capture(fallback_index)
                        if replacement.isOpened():
                            cap.release()
                            cap = replacement
                            current_camera_index = fallback_index
                            state.camera_index = current_camera_index
                            state.camera_name = f"Camera {current_camera_index}"
                            state.last_capture_label = f"camera_{current_camera_index}"
                            state.last_capture_at = time.time()
                            failed_read_count = 0
                            switched = True
                            print(f"AURA TELEMETRY // Auto-switched to camera source {current_camera_index}.")
                            break
                        replacement.release()

                    if switched:
                        continue

                if failed_read_count >= 15:
                    raise RuntimeError("Failed to read from any available camera source.")

                time.sleep(0.2)
                continue

            failed_read_count = 0

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_landmarks(landmark_backend, frame_rgb)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            if isinstance(results, OpenCvLandmarkResult):
                if results.left_eye and results.right_eye:
                    state.face_detected = True
                    state.pose_label = "CENTER (HAAR)"
                    left_eye, right_eye = resolve_eye_points(
                        integration,
                        frame=frame,
                        frame_rgb=frame_rgb,
                        results=results,
                        face_landmarks=None,
                    )
                    score_value = resolve_score_value(integration, results, left_eye, right_eye)
                    mouth_points = results.mouth_points if results.mouth_points is not None else []
                    update_state(state, score_value)
                    if mouth_points:
                        update_mar_state(state, resolve_mar_value(integration, results, mouth_points))
                        draw_mouth_outline(frame, mouth_points, AMBER)
                    if results.face_box is not None:
                        draw_face_box(frame, results.face_box, CYAN)
                        annotate_tracking_label(frame, state.pose_label, results.face_box, CYAN)
                    draw_eye_outline(frame, left_eye, GREEN)
                    draw_eye_outline(frame, right_eye, GREEN)
                else:
                    reset_face_state(state)
            elif has_face_landmarks(results):
                face_landmark_list = results.multi_face_landmarks
                if face_landmark_list is None:
                    reset_face_state(state)
                    continue
                face_landmarks = face_landmark_list[0]
                state.face_detected = True
                state.pose_label = estimate_pose_label(face_landmarks)
                
                # Get pixel coordinates solely for visual canvas overlays
                left_eye, right_eye = resolve_eye_points(
                    integration,
                    frame=frame,
                    frame_rgb=frame_rgb,
                    results=results,
                    face_landmarks=face_landmarks,
                )
                mouth_points = extract_mouth_points(face_landmarks, frame.shape[1], frame.shape[0])
                
                # Get uniform coordinates for aspect-ratio independent biometrics calculations
                left_eye_uni, right_eye_uni = extract_eye_points_uniform(face_landmarks)
                mouth_uni = extract_mouth_points_uniform(face_landmarks)
                
                # Calculate non-distorted biometric ratios
                left_ear = calculate_ear(left_eye_uni)
                right_ear = calculate_ear(right_eye_uni)
                score_value = (left_ear + right_ear) / 2.0
                mar_value = calculate_mar(mouth_uni)
                
                # Update state using robust ratios
                update_state(state, score_value)
                update_mar_state(state, mar_value)
                
                # Render face tracking mesh and biometrics on screen
                face_box = compute_face_box(face_landmarks, frame.shape[1], frame.shape[0])
                draw_face_mesh(frame, face_landmarks)
                draw_face_box(frame, face_box, CYAN)
                annotate_tracking_label(frame, state.pose_label, face_box, CYAN)
                draw_eye_outline(frame, left_eye, GREEN)
                draw_eye_outline(frame, right_eye, GREEN)
                draw_mouth_outline(frame, mouth_points, AMBER)
            else:
                reset_face_state(state)

            snapshot = build_snapshot(state, fps, integration)
            render_dashboard(frame, snapshot)

            # Update live telemetry feeds for web streaming
            try:
                _, jpeg_data = cv2.imencode('.jpg', frame)
                if jpeg_data is not None:
                    with global_telemetry.lock:
                        global_telemetry.latest_frame = jpeg_data.tobytes()
                        global_telemetry.latest_snapshot = snapshot
            except Exception as e:
                pass

            # Check and execute network-driven triggers from HMI
            if global_recalibrate_trigger:
                state.open_reference = 0.0
                state.closed_reference = 0.0
                state.threshold_value = 0.20
                state.calibration_frames = 0
                state.closed_frames = 0
                state.blink_frames = 0
                state.status = "CALIBRATING"
                state.last_capture_label = "recalibrated"
                state.last_capture_at = time.time()
                global_recalibrate_trigger = False
                print("AURA TELEMETRY // Recalibration trigger executed via Web HUD.")
                
            if global_save_open_trigger:
                if isinstance(results, OpenCvLandmarkResult):
                    save_eye_sample(results, "open", state)
                global_save_open_trigger = False
                print("AURA TELEMETRY // Save Open sample executed via Web HUD.")
                
            if global_save_closed_trigger:
                if isinstance(results, OpenCvLandmarkResult):
                    save_eye_sample(results, "closed", state)
                global_save_closed_trigger = False
                print("AURA TELEMETRY // Save Closed sample executed via Web HUD.")

            import sys
            if "--headless" not in sys.argv:
                try:
                    cv2.imshow("Driver Drowsiness Detection", frame)
                    key_code = cv2.waitKey(1) & 0xFF
                    if not handle_keypress(key_code, results, state):
                        break
                except Exception as e:
                    print(f"GUI warning: could not render OpenCV window: {e}. Defaulting to headless mode.")
                    sys.argv.append("--headless")
            else:
                # Limit CPU cycles slightly in headless mode
                time.sleep(0.01)
    finally:
        stop_alarm()
        close_landmark_backend(landmark_backend)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
