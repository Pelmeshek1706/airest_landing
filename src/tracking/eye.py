import numpy as np
import cv2
from tracking.pupil import Pupil


class Eye:
    """
    class representing an eye and providing methods to analyze its landmarks.
    """

    # landmark indices for eye boundaries and iris center (as provided by MediaPipe)
    LEFT_EYE_LANDMARKS = [33, 246, 161, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    RIGHT_EYE_LANDMARKS = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    def __init__(self, frame, landmarks, side):
        """
        initialize the eye instance.

        :param frame: image frame as a numpy array.
        :param landmarks: facial landmarks provided by MediaPipe.
        :param side: 0 for left eye, 1 for right eye.
        """
        self.frame = frame
        self.landmarks = landmarks
        self.side = side  # 0: left, 1: right

        self.center = None         # center point of the eye boundary
        self.pupil = None          # pupil object (based on iris center)
        self.relative_position = None  # stores iris center (in pixel coordinates)

        self._analyze()

    def _analyze(self):
        """
        analyze the eye and iris using provided landmarks.
        """
        if self.side == 0:
            eye_indices = self.LEFT_EYE_LANDMARKS
            iris_index = self.LEFT_IRIS_CENTER
        elif self.side == 1:
            eye_indices = self.RIGHT_EYE_LANDMARKS
            iris_index = self.RIGHT_IRIS_CENTER
        else:
            raise ValueError("side must be 0 (left) or 1 (right)")

        # convert eye boundary landmarks into pixel coordinates
        boundary_pts = np.array([
            (self.landmarks[i].x * self.frame.shape[1],
             self.landmarks[i].y * self.frame.shape[0])
            for i in eye_indices
        ])

        # compute the eye center as the average of boundary points
        center_x = int(np.mean(boundary_pts[:, 0]))
        center_y = int(np.mean(boundary_pts[:, 1]))
        self.center = (center_x, center_y)

        # compute iris center coordinates and create the pupil instance
        iris = self.landmarks[iris_index]
        iris_x = int(iris.x * self.frame.shape[1])
        iris_y = int(iris.y * self.frame.shape[0])
        self.pupil = Pupil(self.frame, (iris_x, iris_y))

        # for further processing, store the iris center position
        self.relative_position = (iris_x, iris_y)

    def map_landmarks_to_coords(self, indices):
        """
        convert landmark indices to pixel coordinates.

        :param indices: list of landmark indices.
        :return: numpy array of shape (n, 2) containing (x, y) coordinates.
        """
        coords = np.array([
            (self.landmarks[i].x * self.frame.shape[1],
             self.landmarks[i].y * self.frame.shape[0])
            for i in indices
        ])
        return coords

    def get_horizontal_ratio(self):
        """
        compute the horizontal ratio of the pupil within the eye.
        The ratio is calculated as:
          (pupil_x - left_boundary) / (right_boundary - left_boundary)
        where a value of 0 indicates extreme right and 1 indicates extreme left.

        :return: horizontal ratio as a float in [0, 1].
        """
        indices = self.LEFT_EYE_LANDMARKS if self.side == 0 else self.RIGHT_EYE_LANDMARKS
        coords = self.map_landmarks_to_coords(indices)
        left_bound = np.min(coords[:, 0])
        right_bound = np.max(coords[:, 0])
        width = right_bound - left_bound

        if width == 0:
            return 0.5  # fallback to center if boundaries are equal
        return (self.pupil.x - left_bound) / width

    def get_vertical_ratio(self):
        """
        compute the vertical ratio of the pupil within the eye.
        The raw ratio is computed as:
          (pupil_y - top_boundary) / (bottom_boundary - top_boundary)
        then an amplification and normalization is applied.

        :return: normalized vertical ratio as a float in [0, 1].
        """
        indices = self.LEFT_EYE_LANDMARKS if self.side == 0 else self.RIGHT_EYE_LANDMARKS
        coords = self.map_landmarks_to_coords(indices)
        top_bound = np.min(coords[:, 1])
        bottom_bound = np.max(coords[:, 1])
        height = bottom_bound - top_bound

        if height == 0:
            return 0.5  # fallback to center if boundaries are equal

        return (self.pupil.y - top_bound) / height

    def get_iris_position(self):
        """
        return the iris center coordinates in pixel space.
        """
        return self.relative_position

    def draw_landmarks(self, frame):
        """
        draw eye boundary landmarks and pupil on the provided frame.
        Landmarks are drawn in blue and the pupil in red.

        :param frame: image frame on which to draw.
        :return: the modified frame.
        """
        indices = self.LEFT_EYE_LANDMARKS if self.side == 0 else self.RIGHT_EYE_LANDMARKS
        coords = self.map_landmarks_to_coords(indices)
        for (x, y) in coords:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        if self.pupil is not None:
            cv2.circle(frame, (self.pupil.x, self.pupil.y), 3, (0, 0, 255), -1)
        return frame

    @staticmethod
    def _amplify_vertical_ratio(raw_ratio, baseline=0.3, factor=1.5):
        """
        amplify the difference between the raw vertical ratio and the baseline.

        :param raw_ratio: the raw vertical ratio.
        :param baseline: baseline ratio when looking straight ahead.
        :param factor: amplification factor.
        :return: amplified ratio clamped to the [0, 1] range.
        """
        amplified = 0.5 + factor * (raw_ratio - baseline)
        return max(0, min(1, amplified))