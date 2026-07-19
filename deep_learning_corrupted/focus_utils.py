import cv2
import mediapipe as mp
import numpy as np

class FocusDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Eye landmarks (MediaPipe Face Mesh)
        # Left Eye (Upper, Lower, Inner, Outer) - approx indices
        # We use a set of landmarks for EAR
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.EAR_THRESHOLD = 0.20 # Below this -> closed

    def calculate_ear(self, landmarks, indices, w, h):
        # indices: [p1, p2, p3, p4, p5, p6]
        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        # p1=362(inner), p4=263(outer) for left eye etc.
        # Actually standard definition matches specific points. 
        # For simplicity with these indices:
        # vertical 1: p2(385) - p6(380)
        # vertical 2: p3(387) - p5(373)
        # horizontal: p1(362) - p4(263)
        
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append(np.array([lm.x * w, lm.y * h]))

        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        hor = np.linalg.norm(coords[0] - coords[3])
        
        if hor == 0: return 0.0
        return (v1 + v2) / (2.0 * hor)

    def process(self, frame):
        """
        Returns:
            is_focused (bool): True if face present AND eyes open.
            reason (str): Reason if not focused.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return False, "Focus"

        # Check Eyes
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        left_ear = self.calculate_ear(face_landmarks, self.LEFT_EYE, w, h)
        right_ear = self.calculate_ear(face_landmarks, self.RIGHT_EYE, w, h)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < self.EAR_THRESHOLD:
            return False, "Focus"

        return True, "Focused"
