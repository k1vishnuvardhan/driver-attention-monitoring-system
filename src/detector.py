import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# Eye and mouth landmark indexes from MediaPipe Face Mesh.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 308, 14, 311]

# Thresholds for detecting closed eyes and yawning.
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
FACE_LANDMARKER_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def euclidean_distance(point1, point2):
    """Return the straight-line distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def get_points(face_landmarks, landmark_indexes, frame_width, frame_height):
    """Convert selected Face Mesh landmarks into pixel coordinates."""
    points = []

    for index in landmark_indexes:
        landmark = face_landmarks.landmark[index]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        points.append((x, y))

    return points


def calculate_ear(
    eye_points=None,
    left_eye=None,
    right_eye=None,
    left_eye_points=None,
    right_eye_points=None,
):
    """
    Calculate Eye Aspect Ratio.

    This function supports both:
    - a single eye passed as `eye_points`
    - both eyes passed as `left_eye` and `right_eye`

    When both eyes are provided, it returns the average EAR so it is compatible
    with `src/main.py`.
    """
    resolved_left = left_eye if left_eye is not None else left_eye_points
    resolved_right = right_eye if right_eye is not None else right_eye_points

    if resolved_left is not None and resolved_right is not None:
        left_value = calculate_ear(eye_points=resolved_left)
        right_value = calculate_ear(eye_points=resolved_right)
        return (left_value + right_value) / 2.0

    if eye_points is None:
        return 0.0

    vertical_1 = euclidean_distance(eye_points[1], eye_points[5])
    vertical_2 = euclidean_distance(eye_points[2], eye_points[4])
    horizontal = euclidean_distance(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_ear(left_eye, right_eye):
    """Return the average EAR from both eyes."""
    return calculate_ear(left_eye=left_eye, right_eye=right_eye)


def eye_aspect_ratio(left_eye, right_eye):
    """Alias kept for compatibility with other modules."""
    return compute_ear(left_eye, right_eye)


def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio.

    Formula:
    MAR = (vertical distance 1 + vertical distance 2) / (2 * horizontal distance)
    """
    vertical_1 = euclidean_distance(mouth_points[1], mouth_points[5])
    vertical_2 = euclidean_distance(mouth_points[2], mouth_points[4])
    horizontal = euclidean_distance(mouth_points[0], mouth_points[3])

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def draw_landmarks(frame, points, color):
    """Draw small circles on selected landmarks."""
    for point in points:
        cv2.circle(frame, point, 3, color, -1)


def create_face_backend():
    """Create a MediaPipe backend that works with the current environment."""
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
        return ("tasks", vision.FaceLandmarker.create_from_options(options))

    raise RuntimeError(
        "No MediaPipe face landmark backend is available. "
        "Expected models/face_landmarker.task for the MediaPipe Tasks Face Landmarker."
    )


def process_face_backend(backend_kind, backend, rgb_frame):
    """Run face landmark detection for the MediaPipe Tasks backend."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return backend.detect_for_video(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))


def extract_face_landmarks(results, backend_kind):
    """Normalize landmark results from the MediaPipe Tasks backend."""
    face_landmarks_list = getattr(results, "face_landmarks", None)
    if face_landmarks_list:
        return face_landmarks_list[0]
    return None


def main():
    backend_kind, backend = create_face_backend()

    # Open the default webcam.
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam.
        ret, frame = cap.read()

        if not ret:
            break

        frame_height, frame_width, _ = frame.shape

        # Convert the frame from BGR to RGB because MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run Face Mesh landmark detection.
        results = process_face_backend(backend_kind, backend, rgb_frame)

        eyes_status = "No Face Detected"
        yawning_status = ""
        eyes_color = (0, 255, 255)
        yawning_color = (255, 255, 255)

        face_landmarks = extract_face_landmarks(results, backend_kind)
        if face_landmarks is not None:

            # Get eye and mouth landmark positions in pixel coordinates.
            left_eye_points = get_points(
                face_landmarks, LEFT_EYE, frame_width, frame_height
            )
            right_eye_points = get_points(
                face_landmarks, RIGHT_EYE, frame_width, frame_height
            )
            mouth_points = get_points(face_landmarks, MOUTH, frame_width, frame_height)

            # Draw eye and mouth landmarks for visualization.
            draw_landmarks(frame, left_eye_points, (0, 255, 0))
            draw_landmarks(frame, right_eye_points, (255, 0, 0))
            draw_landmarks(frame, mouth_points, (0, 0, 255))

            # Calculate average EAR from both eyes.
            average_ear = compute_ear(left_eye_points, right_eye_points)

            # Calculate MAR for yawning detection.
            mar = calculate_mar(mouth_points)

            # Decide whether eyes are open or closed.
            if average_ear < EAR_THRESHOLD:
                eyes_status = "Eyes Closed"
                eyes_color = (0, 0, 255)
            else:
                eyes_status = "Eyes Open"
                eyes_color = (0, 255, 0)

            # Decide whether the person is yawning.
            if mar > MAR_THRESHOLD:
                yawning_status = "Yawning"
                yawning_color = (0, 0, 255)

            # Show EAR and MAR values to help understand the detection.
            cv2.putText(
                frame,
                f"EAR: {average_ear:.2f}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Show eye status on the webcam frame.
        cv2.putText(
            frame,
            eyes_status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            eyes_color,
            2,
        )

        # Show yawning status only when yawning is detected.
        if yawning_status:
            cv2.putText(
                frame,
                yawning_status,
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                yawning_color,
                2,
            )

        # Display the webcam feed.
        cv2.imshow("Driver Drowsiness Detection", frame)

        # Press q to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close the display window.
    cap.release()
    close_method = getattr(backend, "close", None)
    if callable(close_method):
        close_method()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
