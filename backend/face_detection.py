import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp

# Initialize face detector
base_options = BaseOptions(model_asset_path=None)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to access camera")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "Camera Working",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()