import cv2
import mediapipe as mp
from .eye import Eye


class GazeTracking:
    """
    Class that tracks the user's gaze using MediaPipe.
    Provides information about the eye positions, pupil locations,
    and gaze direction (left, right, center, up, or down).
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None

        # initialize MediaPipe face mesh for landmark detection
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    @property
    def pupils_located(self):
        """
        Check that both pupils have been detected.
        :return: True if both left and right pupils are available; otherwise False.
        """
        try:
            return bool(
                self.eye_left and self.eye_left.pupil and
                self.eye_right and self.eye_right.pupil
            )
        except Exception:
            return False

    def refresh(self, frame):
        """
        Update the current frame and process it for gaze tracking.
        
        :param frame: numpy.ndarray representing the image frame.
        """
        self.frame = frame
        self._analyze_frame()

    def _analyze_frame(self):
        """
        Process the current frame by converting it to RGB,
        running MediaPipe face mesh, and initializing Eye objects.
        """
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # use the first detected face
            self.landmarks = results.multi_face_landmarks[0].landmark
            self.eye_left = Eye(self.frame, self.landmarks, side=0)
            self.eye_right = Eye(self.frame, self.landmarks, side=1)
        else:
            self.eye_left = None
            self.eye_right = None

    def pupil_left_coords(self):
        """
        Return the coordinates of the left pupil.
        :return: tuple (x, y) if detected; otherwise None.
        """
        if self.pupils_located:
            return self.eye_left.get_iris_position()
        return None

    def pupil_right_coords(self):
        """
        Return the coordinates of the right pupil.
        :return: tuple (x, y) if detected; otherwise None.
        """
        if self.pupils_located:
            return self.eye_right.get_iris_position()
        return None

    def horizontal_ratio(self):
        """
        Compute the horizontal gaze ratio as the average of both eyes.
        The ratio is 0.0 for extreme right, 0.5 for center, and 1.0 for extreme left.
        
        :return: float in [0, 1] if pupils are detected; otherwise None.
        """
        if self.pupils_located:
            left_ratio = self.eye_left.get_horizontal_ratio()
            right_ratio = self.eye_right.get_horizontal_ratio()
            # left_ratio = self.eye_left.pupil.x / (2 * self.eye_left.center[0])
            # right_ratio = self.eye_right.pupil.x / (2 * self.eye_right.center[0])
            return (left_ratio + right_ratio) / 2
        return None

    def vertical_ratio(self):
        """
        Compute the vertical gaze ratio as the average of both eyes.
        The ratio is 0.0 for extreme top, 0.5 for center, and 1.0 for extreme bottom.
        This implementation uses a simplified calculation.
        
        :return: float in [0, 1] if pupils are detected; otherwise None.
        """
        if self.pupils_located:
            # left_ratio = self.eye_left.get_vertical_ratio()
            # right_ratio = self.eye_right.get_vertical_ratio()
            left_ratio = self.eye_left.pupil.y / (2 * self.eye_left.center[1])
            right_ratio = self.eye_right.pupil.y / (2 * self.eye_right.center[1])
            return (left_ratio + right_ratio) / 2
        return None
    
    def get_head_pitch(self):
        """
        estimate head pitch based on face orientation.
        uses landmarks from MediaPipe's face mesh (landmark 10 for forehead, 152 for chin)
        and returns a pitch offset. When the head is straight, assume the eyes lie at 45% of the face height.
        A positive offset indicates the eyes are lower (head pitched downward) and a negative offset indicates the opposite.
        """
        # convert normalized landmark y-values to pixels
        forehead_y = self.landmarks[10].y * self.frame.shape[0]
        chin_y = self.landmarks[152].y * self.frame.shape[0]
        face_height = chin_y - forehead_y
        if face_height == 0:
            return 0.0

        # use the computed eye centers from the Eye objects (already in pixel coordinates)
        eye_left_y = self.eye_left.center[1]
        eye_right_y = self.eye_right.center[1]
        avg_eye_y = (eye_left_y + eye_right_y) / 2.0

        # normalized eye position between forehead and chin; ideally around 0.45 if head is straight
        normalized_eye_pos = (avg_eye_y - forehead_y) / face_height

        # compute the pitch offset relative to the ideal straight-head value (0.45)
        pitch_offset = normalized_eye_pos - 0.45
        return pitch_offset

    def is_right(self):
        """
        Determine if the user is looking to the right.
        :return: True if the horizontal ratio is less than or equal to 0.35; otherwise False.
        """
        ratio = self.horizontal_ratio()
        return ratio is not None and ratio <= 0.35

    def is_left(self):
        """
        Determine if the user is looking to the left.
        :return: True if the horizontal ratio is greater than or equal to 0.65; otherwise False.
        """
        ratio = self.horizontal_ratio()
        return ratio is not None and ratio >= 0.65

    def is_center(self):
        """
        Determine if the user is looking straight ahead.
        :return: True if the horizontal ratio is between 0.35 and 0.65; otherwise False.
        """
        ratio = self.horizontal_ratio()
        return ratio is not None and 0.35 < ratio < 0.65

    def is_up(self):
        """
        Determine if the user is looking upward.
        :return: True if the vertical ratio is less than or equal to 0.35; otherwise False.
        """
        v_ratio = self.vertical_ratio()
        return v_ratio is not None and v_ratio <= 0.35

    def is_down(self):
        """
        Determine if the user is looking downward.
        :return: True if the vertical ratio is greater than or equal to 0.65; otherwise False.
        """
        v_ratio = self.vertical_ratio()
        return v_ratio is not None and v_ratio >= 0.65

    def is_level(self):
        """
        Determine if the user is looking at eye level.
        :return: True if the vertical ratio is between 0.35 and 0.65; otherwise False.
        """
        v_ratio = self.vertical_ratio()
        return v_ratio is not None and 0.35 < v_ratio < 0.65

    def is_blinking(self):
        """
        Determine if the user is blinking.
        This is based on the ratio of eye width to eye height.
        
        :return: True if blinking ratio exceeds 4.5; otherwise False.
        """
        if self.pupils_located:
            # note: 'blinking' property should be computed in the Eye object
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 4.5
        return False

    def annotated_frame(self):
        """
        Return a copy of the current frame annotated with gaze indicators.
        Draws a green cross at the pupil locations and overlays eye landmarks.
        
        :return: numpy.ndarray representing the annotated frame.
        """
        annotated = self.frame.copy()
        if self.pupils_located:
            color = (0, 255, 0)
            left_coords = self.pupil_left_coords()
            right_coords = self.pupil_right_coords()
            if left_coords:
                x_left, y_left = left_coords
                cv2.line(annotated, (x_left - 5, y_left), (x_left + 5, y_left), color)
                cv2.line(annotated, (x_left, y_left - 5), (x_left, y_left + 5), color)
            if right_coords:
                x_right, y_right = right_coords
                cv2.line(annotated, (x_right - 5, y_right), (x_right + 5, y_right), color)
                cv2.line(annotated, (x_right, y_right - 5), (x_right, y_right + 5), color)
            annotated = self.eye_left.draw_landmarks(annotated)
            annotated = self.eye_right.draw_landmarks(annotated)
        return annotated