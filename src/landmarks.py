from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Standard MediaPipe Face Mesh landmark indices for semantic facial regions
SEMANTIC_LANDMARK_INDICES = {
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
    "nose": [1, 2, 98, 327, 168, 197, 5, 4, 275, 440, 19, 94, 2, 164, 0],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
}

class LandmarkExtractor:
    def __init__(self, static_image_mode=True, max_num_faces=1, refine_landmarks=True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image_input):
        """
        Extracts 2D landmark coordinates (x, y) scaled to image pixel dimensions.
        Accepts a PIL Image, file path (str or Path), or NumPy array (RGB/BGR).
        Returns np.ndarray of shape (N, 2) or None if no face detected.
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
            img_np = np.array(image)
        elif isinstance(image_input, Image.Image):
            img_np = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            raise ValueError("Unsupported image input type.")

        h, w = img_np.shape[:2]
        results = self.face_mesh.process(img_np)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([
            [lm.x * w, lm.y * h] for lm in face_landmarks.landmark
        ], dtype=np.float32)

        return landmarks

    def get_semantic_regions(self, landmarks):
        """
        Extracts landmark point subsets corresponding to specific facial regions.
        """
        if landmarks is None:
            return None

        regions = {}
        for region_name, indices in SEMANTIC_LANDMARK_INDICES.items():
            valid_indices = [i for i in indices if i < len(landmarks)]
            regions[region_name] = landmarks[valid_indices]
        return regions

    def close(self):
        self.face_mesh.close()