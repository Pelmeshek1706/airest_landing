"""
gaze_calibration.py

This module contains the GazeCalibration class for mapping the user's gaze
to screen coordinates. It also includes helper functions to cluster noisy
calibration measurements.
"""

import cv2
import numpy as np
import logging

class GazeCalibration:
    """
    Class for calibrating gaze mapping to screen coordinates.

    This class collects calibration data from a gaze tracking object and computes
    extreme ratios (leftmost/rightmost horizontal and top/bottom vertical) that are
    later used to map raw gaze ratios to actual screen coordinates.
    """

    def __init__(self, gaze_tracking, monitor):
        self.logger = logging.getLogger(__name__)
        self.gaze_tracking = gaze_tracking
        self.test_errors_dict = {'x': [], 'y': [], 'xy': []}

        # initialize extreme ratio values
        self.leftmost_hr = 0
        self.rightmost_hr = 0
        self.top_vr = 0
        self.bottom_vr = 0

        # set up calibration configuration, counters, and timing parameters
        self._init_counters()
        self._init_iris_measurement()
        self._init_timing_parameters()
        self._init_fullscreen_frame(monitor)
        self._init_calibration_grid(monitor)
        self._init_test_points(monitor)

    def _init_calibration_grid(self, monitor):
        self.grid_points = [3, 3]
        self.num_calib_points, self.calibration_points = self._setup_calibration_points(monitor)
        self.calibration_ratios = [[] for _ in range(self.num_calib_points)]

    def _init_test_points(self, monitor):
        self.num_test_points = 5
        self.test_points = self._setup_test_points(monitor)

    def _init_fullscreen_frame(self, monitor):
        self.circle_radius = 20
        self.fullscreen_frame = self._create_fullscreen_frame(monitor)
        self.frame_height, self.frame_width = self.fullscreen_frame.shape[:2]

    def _init_timing_parameters(self):
        self.instruction_frames = 40
        self.fixation_frames = 20
        self.calibration_frames = 40
        self.test_frames = 40

    def _init_counters(self):
        self.current_calib_point = 0
        self.instruction_frame_count = 0
        self.fixation_frame_count = 0
        self.calibration_frame_count = 0
        self.test_point_index = 0
        self.test_frame_count = 0
        self.fl = 0 # for debugging of horizontal/vertical ratios

    def _init_iris_measurement(self):
        self.base_iris_size = 0
        self.iris_size_count = 0
        self.calibration_completed = False
        self.testing_completed = False

    def _setup_calibration_points(self, monitor):
        """
        Compute calibration points evenly distributed over the screen.

        :param monitor: dict with keys 'width' and 'height'
        :return: tuple (number_of_points, list_of_points)
        """
        width = monitor['width']
        height = monitor['height']
        calibration_points = []
        vertical_points, horizontal_points = self.grid_points
        step_x = (width - 2 * self.circle_radius) // (horizontal_points - 1)
        step_y = (height - 2 * self.circle_radius) // (vertical_points - 1)
        for v in range(vertical_points):
            for h in range(horizontal_points):
                x = h * step_x + self.circle_radius
                y = v * step_y + self.circle_radius
                calibration_points.append((x, y))
        return len(calibration_points), calibration_points

    def _setup_test_points(self, monitor):
        """
        Generate random test points on the screen.

        :param monitor: dict with keys 'width' and 'height'
        :return: numpy array of test points.
        """
        width = monitor['width']
        height = monitor['height']
        min_x = self.circle_radius
        max_x = width - self.circle_radius
        min_y = self.circle_radius
        max_y = height - self.circle_radius
        points = [np.array([np.random.randint(min_x, max_x), np.random.randint(min_y, max_y)])
                  for _ in range(self.num_test_points)]
        return np.array(points)

    def _create_fullscreen_frame(self, monitor):
        """
        Create a blank fullscreen frame.

        :param monitor: dict with keys 'width' and 'height'
        :return: numpy array representing the fullscreen frame.
        """
        self.logger.debug(f"Monitor resolution: {monitor['height']} x {monitor['width']}")
        return np.zeros((monitor['height'], monitor['width'], 3), dtype=np.uint8)

    def calibrate_gaze(self):
        """
        Process the calibration state for each frame.

        This method cycles through multiple phases for each calibration point:
          1. Instruction phase
          2. Fixation phase
          3. Data collection phase

        When all calibration points are processed, finalizes calibration.
        """
        self.fullscreen_frame.fill(50)
        if self.current_calib_point < self.num_calib_points:
            self._handle_calibration_phase()
        else:
            self._finalize_calibration()
        return self.fullscreen_frame

    def _handle_calibration_phase(self):
        """
        Handle the current calibration phase:
          - Instruction phase: display instructions.
          - Fixation phase: display the fixation point.
          - Data collection phase: record gaze ratios.
        """
        # Phase 1: instruction phase
        if self.instruction_frame_count < self.instruction_frames:
            self._display_instruction()
            self.instruction_frame_count += 1
            return

        # Phase 2: fixation Phase
        if self.fixation_frame_count < self.fixation_frames:
            self._display_fixation(self.current_calib_point)
            self.fixation_frame_count += 1
            return

        # Phase 3: data collection phase
        if self.calibration_frame_count < self.calibration_frames:
            self._display_fixation(self.current_calib_point)
            self._record_gaze_data(self.current_calib_point)
            self.calibration_frame_count += 1
            return

        # once the data collection phase is over, process and store the collected ratios.
        self.calibration_ratios[self.current_calib_point] = GazeCalibration.cluster_calibration_ratios(
            self.calibration_ratios[self.current_calib_point]
        )
        # reset counters for the next calibration point.
        self.current_calib_point += 1
        self.fixation_frame_count = 0
        self.calibration_frame_count = 0
        self.fl = self.current_calib_point

    def _finalize_calibration(self):
        """
        Once all calibration points are processed, compute the extreme ratios and finalize iris size.
        """
        self._compute_extreme_ratios()
        if self.iris_size_count > 0:
            self.base_iris_size /= self.iris_size_count
        self.calibration_completed = True

    def _display_instruction(self):
        """
        Display calibration instructions on the fullscreen frame.
        """
        cv2.putText(
            self.fullscreen_frame,
            'focus on the red dots',
            (80, 200),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            self.fullscreen_frame,
            'click the window to proceed',
            (80, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def _display_fixation(self, calib_index):
        """
        Display a fixation point (red circle) at the current calibration point.
        """
        cv2.circle(
            self.fullscreen_frame,
            self.calibration_points[calib_index],
            self.circle_radius,
            (0, 0, 255),
            -1
        )

    def _record_gaze_data(self, calib_index):
        """
        Record gaze ratios and (if applicable) iris size for the current calibration point.

        - Records horizontal and vertical ratios if available.
        - If the current calibration point is the center of the screen,
          also record the iris diameter.
        """
        hr = self.gaze_tracking.horizontal_ratio()
        vr = self.gaze_tracking.vertical_ratio()

        if self.fl == self.current_calib_point:
            print('\ncalib p', self.current_calib_point)
            print('hr', hr)
            print('vr', vr)
            self.fl = -1

        if hr is not None and vr is not None:
            self.calibration_ratios[calib_index].append([hr, vr])

        # check if the current calibration point is the center point.
        center_point = (self.frame_width // 2, self.frame_height // 2)
        if self.calibration_points[calib_index] == center_point:
            iris_diameter = self._measure_iris_diameter()
            if iris_diameter:
                self.base_iris_size += iris_diameter
                self.iris_size_count += 1

    def _compute_extreme_ratios(self):
        """
        Compute the extreme horizontal and vertical ratios from the calibration data.

        Iterates over the calibration grid and for each edge point collects the ratio,
        then uses a density-based clustering approach to select the most frequent value.
        """
        vertical_points, horizontal_points = self.grid_points
        left_ratios, right_ratios = [], []
        top_ratios, bottom_ratios = [], []

        for v in range(vertical_points):
            for h in range(horizontal_points):
                idx = v * horizontal_points + h
                ratios = self.calibration_ratios[idx]
                if ratios:
                    # for horizontal ratios, use the first element of the pair.
                    if h == 0:
                        left_ratios.append(ratios[0])
                    elif h == horizontal_points - 1:
                        right_ratios.append(ratios[0])
                    # for vertical ratios, use the second element of the pair.
                    if v == 0:
                        top_ratios.append(ratios[1])
                    elif v == vertical_points - 1:
                        bottom_ratios.append(ratios[1])

        self.leftmost_hr = GazeCalibration.density_cluster_1d(left_ratios)
        self.rightmost_hr = GazeCalibration.density_cluster_1d(right_ratios)
        self.top_vr = GazeCalibration.density_cluster_1d(top_ratios)
        self.bottom_vr = GazeCalibration.density_cluster_1d(bottom_ratios)
        self.logger.debug(
            f"Extreme ratios computed: left {self.leftmost_hr}, right {self.rightmost_hr}, "
            f"top {self.top_vr}, bottom {self.bottom_vr}"
        )

    @staticmethod
    def density_cluster_1d(data):
        """
        Determine the best value from a series of noisy measurements using a density-based approach.
        Computes histograms an array of measurements and selects the bin with the highest frequency as the best estimate

        :param data: list of measurement values.
        :return: best estimated value.
        """
        hist, bins = np.histogram(data, bins='auto')
        idx = int(np.argmax(hist))
        return (bins[idx] + bins[idx + 1]) / 2

    @staticmethod
    def cluster_calibration_ratios(ratios):
        """
        Cluster the calibration ratios for a single calibration point.

        :param ratios: list of [horizontal_ratio, vertical_ratio] pairs.
        :return: list containing [best_horizontal_ratio, best_vertical_ratio].

        This function uses a simple density-based approach to determine the most frequent values.
        It extracts horizontal and vertical ratios from the collected data.
        """
        if not ratios:
            return None
        # extract horizontal and vertical values from each pair.
        horizontal_values = np.array([pair[0] for pair in ratios], dtype=np.float64)
        vertical_values = np.array([pair[1] for pair in ratios], dtype=np.float64)
        best_horizontal = GazeCalibration.density_cluster_1d(horizontal_values)
        best_vertical = GazeCalibration.density_cluster_1d(vertical_values)
        return [best_horizontal, best_vertical]

    def test_gaze(self, pog):
        """
        Display test points and the estimated gaze for evaluation of calibration accuracy.
        """
        self.fullscreen_frame.fill(50)
        if self.test_point_index < self.num_test_points:
            if self.test_frame_count < self.test_frames:
                self._display_test_point(self.test_point_index)
                self._display_estimated_gaze(pog)
                self.test_frame_count += 1
            else:
                # proceed to the next test point and reset the test frame counter.
                self.test_point_index += 1
                self.test_frame_count = 0
        else:
            self.testing_completed = True
            print(f"\nAggregated estimation errors\
                    \nX: {np.mean(np.abs(self.test_errors_dict['x']))} \
                    \nY: {np.mean(np.abs(self.test_errors_dict['y']))} \
                    \nXY: {np.mean(np.abs(self.test_errors_dict['xy']))}\n")
        return self.fullscreen_frame

    def _display_test_point(self, test_index):
        """
        Draw the test point (red circle) on the fullscreen frame.
        """
        cv2.circle(
            self.fullscreen_frame,
            tuple(self.test_points[test_index]),
            self.circle_radius,
            (0, 0, 255),
            -1
        )

    def _display_estimated_gaze(self, pog):
        """
        Obtain the estimated gaze from the PointOfGaze object and display it.

        The gaze is shown as a small light-gray dot.
        """
        est_x, est_y = pog.point_of_gaze()
        if est_x is not None and est_y is not None:
            cv2.circle(
                self.fullscreen_frame,
                (est_x, est_y),
                self.circle_radius // 4,
                (170, 170, 170),
                -1
            )
            # calculate and log error between estimated point and test point
            err_x = est_x - self.test_points[self.test_point_index, 0]
            err_y = est_y - self.test_points[self.test_point_index, 1]
            err_xy = np.linalg.norm(np.asarray([est_x, est_y] - self.test_points[self.test_point_index]))
            self.test_errors_dict['x'].append(err_x)
            self.test_errors_dict['y'].append(err_y)
            self.test_errors_dict['xy'].append(err_xy)

    def _measure_iris_diameter(self):
        """
        Measure the iris diameter using landmarks from both eyes and return the combined diameter.
        """
        iris_diameter_right = self._measure_single_eye_iris(self.gaze_tracking.eye_right, [474, 475, 476, 477])
        iris_diameter_left = self._measure_single_eye_iris(self.gaze_tracking.eye_left, [469, 470, 471, 472])
        return self._combine_iris_diameters(iris_diameter_right, iris_diameter_left)

    def _measure_single_eye_iris(self, eye, iris_indices):
        """
        Measure the iris diameter for a single eye using provided landmark indices.

        :param eye: the eye object containing landmarks.
        :param iris_indices: list of landmark indices for the iris.
        :return: computed iris diameter or None if not available.
        """
        if eye is None or not hasattr(eye, 'landmarks'):
            return None
        landmarks = eye.landmarks
        # calculate width using two opposite iris landmarks.
        width = np.linalg.norm(np.array([
            landmarks[iris_indices[0]].x - landmarks[iris_indices[2]].x,
            landmarks[iris_indices[0]].y - landmarks[iris_indices[2]].y
        ]))
        # calculate height using the other two iris landmarks.
        height = np.linalg.norm(np.array([
            landmarks[iris_indices[1]].x - landmarks[iris_indices[3]].x,
            landmarks[iris_indices[1]].y - landmarks[iris_indices[3]].y
        ]))
        return (width + height) / 2

    def _combine_iris_diameters(self, diameter_right, diameter_left):
        """
        Combine the iris diameters from the right and left eyes.

        - If both measurements are missing, return None.
        - If one measurement is missing, assume the available value applies to both eyes (i.e. multiply by 2).
        - Otherwise, return the average of the two.
        """
        if diameter_right is None and diameter_left is None:
            return None
        if diameter_right is None:
            return diameter_left * 2
        if diameter_left is None:
            return diameter_right * 2
        return (diameter_right + diameter_left) / 2

    def is_completed(self):
        """
        Check if the calibration process is completed.
        """
        return self.calibration_completed

    def is_tested(self):
        """
        Check if the gaze testing phase is completed.
        """
        return self.testing_completed
    