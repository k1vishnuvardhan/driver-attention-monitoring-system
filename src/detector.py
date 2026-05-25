import math
import numpy as np

# Eye and mouth landmark indexes from MediaPipe Face Mesh.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 308, 14, 311]

# Default thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60

def euclidean_distance(point1, point2):
    """Return the straight-line distance between two points."""
    return math.dist(point1, point2)

def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(eye_points) < 6:
        return 0.0

    # Vertical distances
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])
    # Horizontal distance
    h = euclidean_distance(eye_points[0], eye_points[3])

    if h == 0:
        return 0.0

    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio (MAR).
    Formula: MAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Note: For the 6 mouth points, indices are matched to inner lip structure.
    """
    if len(mouth_points) < 6:
        return 0.0

    # Vertical lip opening distances
    v1 = euclidean_distance(mouth_points[1], mouth_points[5])
    v2 = euclidean_distance(mouth_points[2], mouth_points[4])
    # Horizontal width distance
    h = euclidean_distance(mouth_points[0], mouth_points[3])

    if h == 0:
        return 0.0

    return (v1 + v2) / (2.0 * h)

def calculate_ear_uniform(landmarks, indices, width=1000.0, height=1000.0):
    """
    Computes EAR on a uniform coordinate space to eliminate aspect-ratio distortion.
    landmarks: raw normalized face landmarks from MediaPipe solutions.
    """
    points = []
    for idx in indices:
        lm = landmarks[idx] if hasattr(landmarks, '__getitem__') else landmarks.landmark[idx]
        # Project landmarks to a perfectly square virtual space (1000x1000)
        points.append((lm.x * width, lm.y * height))
    return calculate_ear(points)

def calculate_mar_uniform(landmarks, indices, width=1000.0, height=1000.0):
    """
    Computes MAR on a uniform coordinate space to eliminate aspect-ratio distortion.
    """
    points = []
    for idx in indices:
        lm = landmarks[idx] if hasattr(landmarks, '__getitem__') else landmarks.landmark[idx]
        points.append((lm.x * width, lm.y * height))
    return calculate_mar(points)
