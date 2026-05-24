import cv2
import mediapipe as mp


def open_webcam():
    # Try a few camera numbers because the default webcam is not always camera 0.
    for camera_index in [0, 1, 2]:
        # CAP_DSHOW works better for webcams on Windows.
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # Set a normal webcam size.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # A camera can be "opened" but still fail to give a real image.
        success, frame = cap.read()

        if cap.isOpened() and success and frame is not None:
            print(f"Webcam opened using camera index {camera_index}.")
            return cap

        cap.release()

    return None


def main():
    # Initialize MediaPipe Face Mesh.
    # Face Mesh detects many points on the face, such as eyes, lips, nose, and face outline.
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,        # False means we are using video/webcam input.
        max_num_faces=1,                # Detect only one face.
        refine_landmarks=True,          # Gives more accurate landmarks around eyes and lips.
        min_detection_confidence=0.3,   # Lower value makes detection easier for webcams.
        min_tracking_confidence=0.3     # Lower value makes tracking easier after detection.
    )

    # Open the webcam.
    cap = open_webcam()

    # Check if the webcam opened correctly.
    if cap is None:
        print("Error: Could not open webcam.")
        return

    while True:
        # Read one frame from the webcam.
        success, frame = cap.read()

        # If the frame was not read correctly, stop the program.
        if not success:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally so it feels like looking in a mirror.
        frame = cv2.flip(frame, 1)

        # MediaPipe works with RGB images, but OpenCV reads images in BGR format.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # This tells MediaPipe that the image will not be changed while processing.
        # It can make processing faster and more stable.
        rgb_frame.flags.writeable = False

        # Process the frame and detect face landmarks.
        results = face_mesh.process(rgb_frame)

        # Make the image writeable again before drawing on the OpenCV frame.
        rgb_frame.flags.writeable = True

        # If a face is detected, draw the complete facial mesh.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                height, width, _ = frame.shape

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

                for landmark in face_landmarks.landmark:
                    # Landmark coordinates are normalized between 0 and 1.
                    # Convert them to pixel coordinates for drawing on the frame.
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    # Draw a small green circle at each landmark point.
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        else:
            # Show this message when the webcam works but MediaPipe cannot find a face.
            cv2.putText(
                frame,
                "No Face Detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.putText(
                frame,
                "Face camera directly and improve lighting",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Show the webcam frame in a window.
        cv2.imshow("Landmark Detection", frame)

        # Exit the loop when the user presses 'q'.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all OpenCV windows properly.
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
