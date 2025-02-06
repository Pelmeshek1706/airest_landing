import cv2
import mediapipe as mp
from .eye import Eye


class GazeTracking(object):
    """
    This class tracks the user's gaze using MediaPipe.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed.
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None

        # MediaPipe face mesh model for detecting facial and iris landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    @property
    def pupils_located(self):
        """Checks that the pupils have been located"""
        try:
            if self.eye_left.pupil and self.eye_right.pupil:
                return True
            return False
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initializes Eye objects using MediaPipe."""
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark  # Extract the landmarks correctly
            self.eye_left = Eye(self.frame, landmarks, 0)  # Left eye
            self.eye_right = Eye(self.frame, landmarks, 1)  # Right eye
        else:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """
        Refreshes the frame and analyzes it.

        :param frame: (numpy.ndarray) The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            return self.eye_left.get_iris_position()
        return None

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            return self.eye_right.get_iris_position()
        return None

    def horizontal_ratio(self):
        """
        :return: a number between 0.0 and 1.0 that indicates the horizontal direction of the gaze.
        The extreme right is 0.0, the center is 0.5, and the extreme left is 1.0.
        """
        if self.pupils_located:
            hr_left = self.eye_left.get_horizontal_ratio()
            hr_right = self.eye_right.get_horizontal_ratio()
            # print('hr', hr_left, hr_right)
            return (hr_left + hr_right) / 2

    # def vertical_ratio(self):
    #     """
    #     :return: a number between 0.0 and 1.0 that indicates the vertical direction of the gaze.
    #     The extreme top is 0.0, the center is 0.5, and the extreme bottom is 1.0.
    #     """
    #     if self.pupils_located:
    #         vr_left = self.eye_left.get_vertical_ratio()
    #         vr_right = self.eye_right.get_vertical_ratio()
    #         # print('vr', vr_left, vr_right)
    #         return (vr_left + vr_right) / 2

    # def horizontal_ratio(self):
    #     if self.pupils_located:
    #         pupil_left = self.eye_left.pupil.x / (2 * (self.eye_left.center[0]))
    #         pupil_right = self.eye_right.pupil.x / (2 * (self.eye_right.center[0]))
    #         # print('hr', pupil_left, pupil_right)
    #         return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (2 * (self.eye_left.center[1]))
            pupil_right = self.eye_right.pupil.y / (2 * (self.eye_right.center[1]))
            # print('vr', pupil_left, pupil_right)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return 0.35 < self.horizontal_ratio() < 0.65

    def is_up(self):
        """Returns true if the user is looking up"""
        if self.pupils_located:
            return self.vertical_ratio() <= 0.35

    def is_down(self):
        """Returns true if the user is looking down"""
        if self.pupils_located:
            return self.vertical_ratio() >= 0.65

    def is_level(self):
        """Returns true if the user is looking at eye level"""
        if self.pupils_located:
            return 0.35 < self.vertical_ratio() < 0.65

    def is_blinking(self):
        """Returns true if the user closes his eyes. Is based on eye_width / eye_height"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 4.5

    def annotated_frame(self):
        """Returns the main frame with pupils marked with a green cross"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            frame = self.eye_left.draw_landmarks(frame)
            frame = self.eye_right.draw_landmarks(frame)
        return frame