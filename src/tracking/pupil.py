import cv2

class Pupil(object):
    """
    This class represents the pupil position based on the iris center provided by MediaPipe.
    """

    def __init__(self, iris_frame, iris_center):
        """
        initializes the Pupil object with the iris frame and center coordinates.

        :param iris_frame: (numpy.ndarray) Frame containing the iris region.
        :param iris_center: (tuple) The (x, y) coordinates of the iris center.
        """
        self.iris_frame = iris_frame
        self.x = iris_center[0]
        self.y = iris_center[1]

        # Visualize the pupil detection for debugging
        self._visualize_pupil()

    def _visualize_pupil(self):
        """
        visualizes the detected pupil position on the iris frame.
        """
        # cv2.circle(self.iris_frame, (self.x, self.y), 2, (0, 255, 0), -1)
        # cv2.imshow('Detected Pupil', self.iris_frame)
        # cv2.waitKey(1)

    def get_pupil_coords(self):
        """
        returns the coordinates of the detected pupil center.

        :return: (x, y) coordinates of the pupil.
        """
        return self.x, self.y