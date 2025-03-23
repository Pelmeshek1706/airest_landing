"""
gaze_calibration.py

This module contains the GazeCalibration class for mapping the user's gaze
to screen coordinates. It also includes helper functions to cluster noisy
calibration measurements.
"""

import cv2
import numpy as np
import logging
from distortion_calibration import compute_distortion_coefficients

class GazeCalibration:
    """
    Class for calibrating gaze mapping to screen coordinates.

    This class collects calibration data from a gaze tracking object and computes
    extreme ratios (leftmost/rightmost horizontal and top/bottom vertical) that are
    later used to map raw gaze ratios to actual screen coordinates.
    """

    def __init__(self, gaze_tracking, monitor, video_source, record):
        self.logger = logging.getLogger(__name__)
        self.gaze_tracking = gaze_tracking
        if video_source or record:
            debug_errors = True
        self.errors_dict = {'x': [], 'y': [], 'xy': []}

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
        self._init_test_points(monitor, debug_errors)

    def _init_calibration_grid(self, monitor):
        self.grid_points = [3, 3]
        self.num_calib_points, self.calibration_points = self._setup_calibration_points(monitor)
        self.calibration_ratios = [[] for _ in range(self.num_calib_points)]
        self.calibration_data = [[] for _ in range(self.num_calib_points)]

    def _init_test_points(self, monitor, debug_errors):
        if debug_errors:
            self.test_points = self._generate_test_points(monitor)
            self.num_test_points = len(self.test_points)
        else:
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
        return len(calibration_points), np.array(calibration_points)

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
        test_points = [np.array([np.random.randint(min_x, max_x), np.random.randint(min_y, max_y)])
                  for _ in range(self.num_test_points)]
        return np.array(test_points)

    def _generate_test_points(self, monitor, rows=3, cols=3, margin_ratio=0.1):
        """
        deterministically generate test points that are scattered across the screen
        but not at the extreme edges.
        
        :param monitor: dict with keys 'width' and 'height'
        :param rows: number of rows in the grid
        :param cols: number of columns in the grid
        :param margin_ratio: fraction of width/height to leave as margin from the edges
        :return: numpy array of test points (each as (x, y))
        """
        width = monitor['width']
        height = monitor['height']
        margin_x = margin_ratio * width
        margin_y = margin_ratio * height

        # available space after margins
        available_width = width - 2 * margin_x
        available_height = height - 2 * margin_y

        # compute equally spaced x and y values within the available space
        xs = [int(margin_x + i * (available_width / (cols - 1))) for i in range(cols)]
        ys = [int(margin_y + i * (available_height / (rows - 1))) for i in range(rows)]

        # generate points as all combinations of xs and ys
        points = [(x, y) for y in ys for x in xs]
        return np.array(points)

    def _create_fullscreen_frame(self, monitor):
        """
        Create a blank fullscreen frame.

        :param monitor: dict with keys 'width' and 'height'
        :return: numpy array representing the fullscreen frame.
        """
        self.logger.debug(f"Monitor resolution: {monitor['height']} x {monitor['width']}")
        return np.zeros((monitor['height'], monitor['width'], 3), dtype=np.uint8)

    def calibrate_gaze(self, pog):
        """
        Process the calibration state for each frame.

        This method cycles through multiple phases for each calibration point:
          1. Instruction phase
          2. Fixation phase
          3. Data collection phase

        When all calibration points are processed, finalizes calibration.
        """
        self.pog = pog
        self.fullscreen_frame.fill(255)
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
        
        # save final ratios for the current point
        self.calibration_data[self.current_calib_point] = (*self.calibration_ratios[self.current_calib_point],
                                                           *self.calibration_points[self.current_calib_point])

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
        if self.iris_size_count > 0: # if no iris measurements were recorded in that frame
            self.base_iris_size /= self.iris_size_count
        self.calibration_completed = True
        self.compute_polynomial_mapping()

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
            (25),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            self.fullscreen_frame,
            'click the window to proceed',
            (80, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            (25),
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

    def _update_iris_size(self, iris_d):
        self.base_iris_size += iris_d
        self.iris_size_count += 1

    def _record_gaze_data(self, calib_index):
        """
        Record gaze ratios and (if applicable) iris size for the current calibration point.

        - Records horizontal and vertical ratios if available.
        - If the current calibration point is the center of the screen,
          also record the iris diameter.
        """
        hr = self.gaze_tracking.horizontal_ratio()
        vr = self.gaze_tracking.vertical_ratio()
        
        if hr is not None and vr is not None:
            self.calibration_ratios[calib_index].append([hr, vr])

        # check if the current calibration point is the center point.
        center_point = [self.frame_width // 2, self.frame_height // 2]
        if (self.calibration_points[calib_index] == center_point).all():
            iris_diameter = self._measure_iris_diameter()
            if iris_diameter:
                self._update_iris_size(iris_diameter)

        if self.fl == self.current_calib_point:
            print('\ncalib p', self.current_calib_point)
            print('hr', hr)
            print('vr', vr)
            self.fl = -1

    def _compute_extreme_ratios(self):
        """
        Compute the extreme horizontal and vertical ratios from the calibration data.
        
        This version precomputes the indices for the edges of the calibration grid
        and collects the corresponding ratios using list comprehensions.
        """
        vertical_points, horizontal_points = self.grid_points

        # compute indices for each edge:
        left_edge_indices = [v * horizontal_points for v in range(vertical_points)]
        right_edge_indices = [v * horizontal_points + (horizontal_points - 1) for v in range(vertical_points)]
        top_edge_indices = list(range(horizontal_points))
        bottom_edge_indices = list(range((vertical_points - 1) * horizontal_points, vertical_points * horizontal_points))
        
        # gather the ratios (if a given calibration point exists, i.e. non-empty)
        left_ratios = [self.calibration_ratios[idx][0] for idx in left_edge_indices
                       if self.calibration_ratios[idx]]
        right_ratios = [self.calibration_ratios[idx][0] for idx in right_edge_indices
                        if self.calibration_ratios[idx]]
        top_ratios = [self.calibration_ratios[idx][1] for idx in top_edge_indices
                      if self.calibration_ratios[idx]]
        bottom_ratios = [self.calibration_ratios[idx][1] for idx in bottom_edge_indices
                         if self.calibration_ratios[idx]]
        
        # Fallbacks in case any list is empty (adjust as needed)
        if not left_ratios:
            left_ratios = [0.0]
        if not right_ratios:
            right_ratios = [1.0]
        if not top_ratios:
            top_ratios = [0.0]
        if not bottom_ratios:
            bottom_ratios = [1.0]
        
        # cluster the collected ratios using the density-based approach
        self.leftmost_hr = GazeCalibration.density_cluster_1d(left_ratios)
        self.rightmost_hr = GazeCalibration.density_cluster_1d(right_ratios)
        self.top_vr = GazeCalibration.density_cluster_1d(top_ratios)
        self.bottom_vr = GazeCalibration.density_cluster_1d(bottom_ratios)
        
        self.logger.debug(
            f"Extreme ratios computed: left {self.leftmost_hr}, right {self.rightmost_hr}, "
            f"top {self.top_vr}, bottom {self.bottom_vr}"
        )

    def compute_polynomial_mapping(self):
        # Assume self.calibration_data is a list of tuples [(hr, vr, x, y), ...]
        data = np.array(self.calibration_data)
        hr = data[:, 0]
        vr = data[:, 1]
        x_values = data[:, 2]
        y_values = data[:, 3]
        
        # Build design matrix: [hr^2, vr^2, hr*vr, hr, vr, 1]
        A = np.column_stack([hr**2, vr**2, hr*vr, hr, vr, np.ones_like(hr)])
        
        # Solve least squares for x mapping and y mapping
        self.poly_x = np.linalg.lstsq(A, x_values, rcond=None)[0]
        self.poly_y = np.linalg.lstsq(A, y_values, rcond=None)[0]

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
        process the test phase for each frame.
        
        this method displays test points, the estimated gaze,
        and records errors for evaluation of calibration accuracy.
        once all test points have been processed, it prints aggregated errors.
        
        :param pog: the PointOfGaze object used for gaze estimation.
        :return: the fullscreen frame with test overlays.
        """
        self.pog = pog
        self.fullscreen_frame.fill(255)
        
        if self.test_point_index < self.num_test_points:
            self._handle_test_phase()
        else:
            self.testing_completed = True
            self.print_aggregated_errors()
        
        return self.fullscreen_frame

    def _handle_test_phase(self):
        """
        handle the test phase for the current test point.
        
        during the test phase, display the test point and estimated gaze
        for a fixed number of frames. after that, record the error for the 
        current test point, advance to the next test point, and reset the frame count.
        """
        if self.test_frame_count < self.test_frames:
            self._display_test_point(self.test_point_index)
            self._display_estimated_gaze()  # this method draws the gaze dot and logs errors if needed
            self.test_frame_count += 1
        else:
            self.test_point_index += 1
            self.test_frame_count = 0

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

    def _display_estimated_gaze(self):
        """
        Obtain the estimated gaze from the PointOfGaze object and display it.

        The gaze is shown as a small light-gray dot.
        """
        est_x, est_y = self._record_error()

        if est_x is not None and est_y is not None:
            cv2.circle(
                self.fullscreen_frame,
                (est_x, est_y),
                self.circle_radius // 4,
                (170, 170, 170),
                -1
            )

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
    
    def _calculate_error(self, screen_x, screen_y):
        """
        calculate and log the error between the estimated gaze coordinates and a target point.
        
        :param screen_x: estimated gaze x coordinate.
        :param screen_y: estimated gaze y coordinate.
        :return: tuple (err_x, err_y, err_xy) representing the error in x, error in y, and Euclidean error.
        """
        target_point = self.test_points[self.test_point_index]
        target_point = np.asarray(target_point)
        
        err_x = screen_x - target_point[0]
        err_y = screen_y - target_point[1]
        err_xy = np.linalg.norm(np.array([screen_x, screen_y]) - target_point)
        
        self.errors_dict['x'].append(err_x)
        self.errors_dict['y'].append(err_y)
        self.errors_dict['xy'].append(err_xy)
        
        # print(f"Error calculated: err_x={err_x}, err_y={err_y}, err_xy={err_xy}")
            
    def _record_error(self):
        """
        display the estimated gaze and record the error relative to a target point.
        
        the function draws a small light-gray dot at the estimated gaze and then calls
        calculate_error() to compute and record the error.
        """
        est_x, est_y = self.pog.point_of_gaze()
        if est_x is None or est_y is None:
            return  # nothing to display or record
        
        # record the error using the common calculation function
        # this function will use self.test_points (or self.calibration_points) based on mode
        self._calculate_error(est_x, est_y)

        return est_x, est_y

    def print_aggregated_errors(self):
        """
        print the aggregated estimation errors for the given mode.
        """

        mean_x = np.mean(np.abs(self.errors_dict['x']))
        mean_y = np.mean(np.abs(self.errors_dict['y']))
        mean_xy = np.mean(np.abs(self.errors_dict['xy']))

        print(f"\nAggregated estimation errors:\n"
                f"X: {mean_x}\n"
                f"Y: {mean_y}\n"
                f"XY: {mean_xy}\n")
