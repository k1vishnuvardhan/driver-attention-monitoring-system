import cv2
import mediapipe as mp
import numpy as np


# Eye and mouth landmark indexes from MediaPipe Face Mesh.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 308, 14, 311]

# Thresholds for detecting closed eyes and yawning.
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6


def open_webcam():
    """Open the first working webcam."""
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if cap.isOpened():
            print(f"Webcam opened using camera index {camera_index}.")
            return cap

        cap.release()

    return None


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


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio.

    Formula:
    EAR = (vertical distance 1 + vertical distance 2) / (2 * horizontal distance)
    """
    vertical_1 = euclidean_distance(eye_points[1], eye_points[5])
    vertical_2 = euclidean_distance(eye_points[2], eye_points[4])
    horizontal = euclidean_distance(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


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


def draw_full_face_mesh(frame, face_landmarks, mp_face_mesh, mp_drawing, mp_styles):
    """Draw the complete MediaPipe face mesh on the frame."""
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
    )


def main():
    # Start MediaPipe Face Mesh. It detects 468 face landmarks.
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open the webcam.
    cap = open_webcam()

    if cap is None:
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam.
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame so it looks like a mirror.
        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape

        # Convert the frame from BGR to RGB because MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run Face Mesh landmark detection.
        results = face_mesh.process(rgb_frame)

        eyes_status = "No Face Detected"
        yawning_status = ""
        eyes_color = (0, 255, 255)
        yawning_color = (255, 255, 255)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Draw the full facial mesh instead of only the eye and mouth points.
            draw_full_face_mesh(
                frame, face_landmarks, mp_face_mesh, mp_drawing, mp_styles
            )

            # Get eye and mouth landmark positions in pixel coordinates.
            left_eye_points = get_points(
                face_landmarks, LEFT_EYE, frame_width, frame_height
            )
            right_eye_points = get_points(
                face_landmarks, RIGHT_EYE, frame_width, frame_height
            )
            mouth_points = get_points(face_landmarks, MOUTH, frame_width, frame_height)

            # Highlight the landmarks used for EAR and MAR calculations.
            draw_landmarks(frame, left_eye_points, (0, 255, 0))
            draw_landmarks(frame, right_eye_points, (255, 0, 0))
            draw_landmarks(frame, mouth_points, (0, 0, 255))

            # Calculate average EAR from both eyes.
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            average_ear = (left_ear + right_ear) / 2.0

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
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
