import cv2
import os
from .config import CASCADE_FILENAME

class FaceDetector:
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            # Try to load from cv2 data
            cascade_path = os.path.join(cv2.data.haarcascades, CASCADE_FILENAME)
        
        if not os.path.exists(cascade_path):
             # Fallback to local file if it exists, or raise warning
             if os.path.exists(CASCADE_FILENAME):
                 cascade_path = CASCADE_FILENAME
             else:
                 print(f"Warning: Haar Cascade not found at {cascade_path}")

        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        """
        Detects faces in an image.
        Returns a list of tuples (x, y, w, h).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ScaleFactor=1.1, minNeighbors=5, minSize=(30, 30) for standard face detection
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return rects
