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

import cv2
import mediapipe as mp
import numpy as np

from dashboard import DashboardSnapshot, render_dashboard


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
MIN_BLINK_FRAMES = 2
WARNING_FRAMES = 15
DROWSY_FRAMES = 30
CALIBRATION_FRAMES = 45

GREEN = (60, 179, 113)
ALARM_PATH = Path(__file__).resolve().parent.parent / "assets" / "alarm.wav"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OPEN_DATA_DIR = DATA_ROOT / "open"
CLOSED_DATA_DIR = DATA_ROOT / "closed"
FACE_LANDMARKER_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


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


def calculate_ear(eye_points: list[tuple[int, int]]) -> float:
    vertical_1 = euclidean_distance(eye_points[1], eye_points[5])
    vertical_2 = euclidean_distance(eye_points[2], eye_points[4])
    horizontal = euclidean_distance(eye_points[0], eye_points[3])
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


def draw_eye_outline(frame, eye_points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for point in eye_points:
        cv2.circle(frame, point, 2, color, -1)
    contour = np.array(eye_points, dtype=np.int32)
    cv2.polylines(frame, [cv2.convexHull(contour)], True, color, 1)


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
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
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
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
        timestamp_ms = int(time.time() * 1000)
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

    return OpenCvLandmarkResult(
        left_eye=left_eye,
        right_eye=right_eye,
        left_eye_crop=left_crop,
        right_eye_crop=right_crop,
        score=score,
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


def build_snapshot(state: DetectorState, fps: float, integration: IntegrationContext) -> DashboardSnapshot:
    return DashboardSnapshot(
        status=state.status,
        score_value=state.current_score,
        threshold_value=state.threshold_value,
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
    )


def reset_face_state(state: DetectorState) -> None:
    state.status = "NO FACE"
    state.closed_frames = 0
    state.blink_frames = 0
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
    ensure_data_dirs()
    integration = load_integration_context()
    landmark_backend = create_landmark_backend(integration)
    if integration.mode == "internal":
        integration.mode = landmark_backend.name

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    state = DetectorState()
    previous_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_landmarks(landmark_backend, frame_rgb)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            if isinstance(results, OpenCvLandmarkResult):
                if results.left_eye and results.right_eye:
                    left_eye, right_eye = resolve_eye_points(
                        integration,
                        frame=frame,
                        frame_rgb=frame_rgb,
                        results=results,
                        face_landmarks=None,
                    )
                    score_value = resolve_score_value(integration, results, left_eye, right_eye)
                    update_state(state, score_value)
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
                left_eye, right_eye = resolve_eye_points(
                    integration,
                    frame=frame,
                    frame_rgb=frame_rgb,
                    results=results,
                    face_landmarks=face_landmarks,
                )
                score_value = resolve_score_value(integration, results, left_eye, right_eye)
                update_state(state, score_value)
                draw_eye_outline(frame, left_eye, GREEN)
                draw_eye_outline(frame, right_eye, GREEN)
            else:
                reset_face_state(state)

            render_dashboard(frame, build_snapshot(state, fps, integration))
            cv2.imshow("Driver Drowsiness Detection", frame)

            key_code = cv2.waitKey(1) & 0xFF
            if not handle_keypress(key_code, results, state):
                break
    finally:
        stop_alarm()
        close_landmark_backend(landmark_backend)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
