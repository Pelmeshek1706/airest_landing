import numpy as np
import cv2
from .pupil import Pupil

class Eye(object):
    # Example indices for eye landmarks and iris center in MediaPipe
    LEFT_EYE_LANDMARKS = [33, 246, 161, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  # Example indices for left eye boundary
    RIGHT_EYE_LANDMARKS = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]  # Example indices for right eye boundary
    LEFT_IRIS_CENTER = 468  # Example landmark index for left iris center in MediaPipe
    RIGHT_IRIS_CENTER = 473  # Example landmark index for right iris center in MediaPipe

    def __init__(self, frame, landmarks, side):
        self.frame = frame
        self.landmarks = landmarks
        self.side = side

        self.center = None
        self.pupil = None
        self.blinking = None

        self._analyze()

    def _analyze(self):
        """
        Analyzes the eye and iris using landmarks provided by MediaPipe.
        """
        if self.side == 0:  # Left eye
            eye_landmarks = self.LEFT_EYE_LANDMARKS
            iris_center_index = self.LEFT_IRIS_CENTER
        elif self.side == 1:  # Right eye
            eye_landmarks = self.RIGHT_EYE_LANDMARKS
            iris_center_index = self.RIGHT_IRIS_CENTER
        else:
            return

        # Calculate the eye center by averaging key points of the eye boundary
        eye_points = np.array([
            (self.landmarks[i].x * self.frame.shape[1], self.landmarks[i].y * self.frame.shape[0])
            for i in eye_landmarks
        ])

        # Calculate the mean of the x and y coordinates
        eye_center_x = int(np.mean(eye_points[:, 0]))
        eye_center_y = int(np.mean(eye_points[:, 1]))

        self.center = (eye_center_x, eye_center_y)

        # Get the iris center coordinates
        iris_center = self.landmarks[iris_center_index]
        iris_center_x = int(iris_center.x * self.frame.shape[1])
        iris_center_y = int(iris_center.y * self.frame.shape[0])

        # Store the iris center as pupil coordinates for simplicity
        self.pupil = Pupil(self.frame, (iris_center_x, iris_center_y))

        # Calculate relative position of the iris center to the eye center
        relative_iris_x = iris_center_x
        relative_iris_y = iris_center_y

        # Store the relative position for further processing or debugging
        self.relative_position = (relative_iris_x, relative_iris_y)

    def get_horizontal_ratio(self):
        """
        Returns the horizontal ratio for this eye computed as:
        (pupil_x - left_boundary) / (right_boundary - left_boundary)
        """
        landmarks_idx = self.LEFT_EYE_LANDMARKS if self.side == 0 else self.RIGHT_EYE_LANDMARKS
        eye_points = self.map_landmarks_coords(landmarks_idx)
        # Determine the leftmost and rightmost x coordinates:
        E_l = np.min(eye_points[:, 0])
        E_r = np.max(eye_points[:, 0])
        # Protect against division by zero
        if (E_r - E_l) == 0:
            return 0.5  # fallback to center
        return (self.pupil.x - E_l) / (E_r - E_l)
    
    def get_vertical_ratio(self):
        """
        Returns the vertical ratio for this eye computed as:
        (pupil_y - top_boundary) / (bottom_boundary - top_boundary)
        """
        landmarks_idx = self.LEFT_EYE_LANDMARKS if self.side == 0 else self.RIGHT_EYE_LANDMARKS
        eye_points = self.map_landmarks_coords(landmarks_idx)
        # Concatenate both groups’ y-coordinates
        E_t = np.min(eye_points[:, 1])
        E_b = np.max(eye_points[:, 1])
        if (E_b - E_t) == 0:
            return 0.5
        raw_vr = (self.pupil.y - E_t) / (E_b - E_t)
        amp_vr = amplify_vertical_ratio(raw_vr)
        vr_norm = (raw_vr - 0.30) / (0.40 - 0.30)
        print(f'{"left" if self.side == 0 else "right"}', raw_vr)
        return vr_norm

    def map_landmarks_coords(self, landmarks_idx):
        eye_points = np.array([
            (self.landmarks[i].x * self.frame.shape[1], self.landmarks[i].y * self.frame.shape[0])
            for i in landmarks_idx])
        return eye_points # shape: (m, 2) - (num of indices, x/y coords)

    def get_iris_position(self):
        """Returns the iris center coordinates relative to the eye center."""
        return self.relative_position
    
    def draw_landmarks(self, frame):
        """
        Draws circles on the frame for the horizontal and vertical landmarks,
        as well as the pupil.
        Horizontal landmarks will be drawn in green, vertical in blue,
        and the pupil in red.
        """
        # Determine which dictionaries to use for this eye:
        if self.side == 0:
            landmarks_idx = self.LEFT_EYE_LANDMARKS
        else:
            landmarks_idx = self.RIGHT_EYE_LANDMARKS

        # Draw landmarks in blue
        eye_points = self.map_landmarks_coords(landmarks_idx)
        for i, (x, y) in enumerate(eye_points):
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1) # blue - top, pink - bottom

        # Draw the pupil in red
        if self.pupil is not None:
            cv2.circle(frame, (self.pupil.x, self.pupil.y), 3, (0, 0, 255), -1)

        return frame
    
@staticmethod
def amplify_vertical_ratio(raw_vr, baseline_vr=0.3, amplification_factor=1.5):
    """
    Amplifies the difference between raw_vr and baseline_vr.

    Args:
        raw_vr: The raw vertical ratio computed from the landmarks.
        baseline_vr: The baseline vertical ratio when looking straight ahead.
        amplification_factor: A factor to amplify the difference.

    Returns:
        A vertical ratio in the [0, 1] range after amplification.
    """
    amplified = 0.5 + amplification_factor * (raw_vr - baseline_vr)
    # Clamp the value between 0 and 1
    amplified = max(0, min(1, amplified))
    return amplified
