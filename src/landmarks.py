import cv2
import mediapipe as mp

class FaceMeshWrapper:
    def __init__(self, max_faces=1, refine=True, det_confidence=0.5, track_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=det_confidence,
            min_tracking_confidence=track_confidence
        )

    def get_landmarks(self, frame):
        """
        Processes frame and returns a list of (x, y) landmark pixel coordinates.
        Returns None if no face is detected.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            points = []
            for lm in landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
            return points
        return None

    def close(self):
        self.face_mesh.close()
