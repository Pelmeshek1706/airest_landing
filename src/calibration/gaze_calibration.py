"""
gaze_calibration.py

This module contains the GazeCalibrationAPI class for mapping the user's gaze
to screen coordinates. It splits the calibration process into three stages:
  1. Ratios Calibration Stage (collecting raw eye ratios)
  2. Distortion Calibration Stage (deriving mapping/distortion coefficients)
  3. Test Stage (final error estimation)
These sub‐classes are encapsulated within GazeCalibrationAPI so that external
dependencies (in PointOfGaze and EPOGAnalyzer) remain unchanged.
"""

import cv2
import logging
import numpy as np
from scipy.optimize import least_squares

COLOR_SCHEME = {'LIGHT': {'bg':255, 'text':(0,0,0)},
                'DARK': {'bg':0, 'text':(255, 255, 255)}}
COLOR = COLOR_SCHEME['LIGHT']

class GazeCalibration:
    """
    Class for calibrating gaze mapping to screen coordinates.

    This class collects calibration data from a gaze tracking object and computes
    extreme ratios (leftmost/rightmost horizontal and top/bottom vertical) that are
    later used to map raw gaze ratios to actual screen coordinates.
    """

    ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ###
    # --- Initialization and Configuration --- #

    def __init__(self, gaze_tracking, monitor):
        self.logger = logging.getLogger(__name__)
        self.gaze_tracking = gaze_tracking
        self.monitor = monitor

        self.errors_dict = {'x': [], 'y': [], 'xy': []}

        # set up calibration configuration, counters, and timing parameters
        self._init_counters()
        self._init_iris_measurement()
        self._init_timing_parameters()
        self._init_fullscreen_frame()
        self._init_calibration_points(extreme_points=True)
        
        # this list will store calibration data for each calibration point.
        # each element is a tuple: (hr, vr, target_x, target_y)
        self.calibration_ratios = [[] for _ in range(self.num_calibration_points)]
        self.calibration_data = [[] for _ in range(self.num_calibration_points)]
        # distortion_data stores a list of raw (x,y) estimates for each calibration point
        self.distortion_data = [[] for _ in range(self.num_calibration_points)]

        # new: set initial calibration stage to "ratios"
        print('\n--- Ratios Calibration Stage ---\n')
        self.calibration_stage = "ratios"  # stages: "ratios" -> "distortion" -> "test"
        self.calibration_completed = [False, False, False]  # flags for each stage

        # These will be computed during distortion calibration:
        self.poly_x = None
        self.poly_y = None
        self.distortion_params = None

    def _init_counters(self):
        self.current_calibration_point = 0
        self.instruction_frame_count = 0
        self.fixation_frame_count = 0
        self.calibration_frame_count = 0
        self.test_point_index = 0
        self.test_frame_count = 0
        self.fl = 0 # for debugging of horizontal/vertical ratios

    def _init_iris_measurement(self):
        self.base_iris_size = 0
        self.iris_size_count = 0

    def _init_timing_parameters(self):
        self.instruction_frames = 40
        self.fixation_frames = 20
        self.calibration_frames = 40
        self.test_frames = 40

    def _init_fullscreen_frame(self):
        self.circle_radius = 20
        self.fullscreen_frame = np.zeros((self.monitor['height'], self.monitor['width'], 3), dtype=np.uint8)
        self.frame_height, self.frame_width = self.fullscreen_frame.shape[:2]

    def _init_calibration_points(self, extreme_points=False):
        self.calibration_points = self._generate_calibration_points(extreme_points=extreme_points)
        self.num_calibration_points = len(self.calibration_points)

    def _generate_calibration_points(self, rows=3, cols=3, margin_ratio=0.1, extreme_points=False):
        """
        deterministically generate test points that are scattered across the screen
        but not at the extreme edges.
        
        :param rows: number of rows in the grid
        :param cols: number of columns in the grid
        :param margin_ratio: fraction of width/height to leave as margin from the edges
        :return: numpy array of test points (each as (x, y))
        """
        width = self.monitor['width']
        height = self.monitor['height']

        if extreme_points:
            margin = self.circle_radius
            xs = [int(margin + i * ((width - 2 * margin) / (cols - 1))) for i in range(cols)]
            ys = [int(margin + i * ((height - 2 * margin) / (rows - 1))) for i in range(rows)]

        else:
            margin_x = margin_ratio * width
            margin_y = margin_ratio * height
            available_width = width - 2 * margin_x
            available_height = height - 2 * margin_y
            xs = [int(margin_x + i * available_width / (cols - 1)) for i in range(cols)]
            ys = [int(margin_y + i * available_height / (rows - 1)) for i in range(rows)]

        points = [(x, y) for y in ys for x in xs]
        return np.array(points)



    ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ###
    # --- Main API Methods --- #

    def calibrate_gaze(self, pog):
        """
        Process the calibration stage for each frame.
        Behavior changes based on self.stage:
          - "ratios": collect raw ratios.
          - "distortion": collect raw (x,y) estimates.
          - "test": run the test stage.
        """
        self.pog = pog
        self.fullscreen_frame.fill(COLOR['bg'])

        if self.calibration_stage == "ratios":
            self._handle_stage_phase(self._record_ratios, self._process_ratios)

            if self.current_calibration_point >= self.num_calibration_points:
                self.logger.debug("Ratios Calibration complete. Switching to Distortion Calibration Stage.")

                # Reset counters for distortion stage:
                self.current_calibration_point = 0
                self.instruction_frame_count = 0
                self.fixation_frame_count = 0
                self.calibration_frame_count = 0

                self.compute_polynomial_mapping()  # Use ratios data for initial mapping.

                self.calibration_completed[0] = True
                self.calibration_stage = "distortion"
                self._init_calibration_points(extreme_points=False)

        elif self.calibration_stage == "distortion":
            self._handle_stage_phase(self._record_distortion, self._process_distortion)

            if self.current_calibration_point >= self.num_calibration_points:
                print('\n--- Distortion Calibration Stage ---\n')
                self.logger.debug("Distortion Calibration complete. Switching to Test Stage.")

                # Reset counters for test stage:
                self.current_calibration_point = 0
                self.instruction_frame_count = 0
                self.calibration_frame_count = 0

                # distortion_data is now a list of tuples [(x_raw_avg, y_raw_avg, target_x, target_y), ...]
                self.distortion_params = self.compute_distortion_coefficients()
                print('Distortion params:', self.distortion_params)
                self.logger.debug("Distortion coefficients computed: {}".format(self.distortion_params))
                
                print('\nNumber of clusters created:', self.pog.get_num_clusters()) # FOR DEBUGGING
                self.pog.num_clusters = 0 # FOR DEBUGGING
                self.print_aggregated_errors()

                self.calibration_completed[1] = True
                self.calibration_stage = "test"
                self._init_calibration_points(extreme_points=False)

        elif self.calibration_stage == "test":
            self._handle_stage_phase(self._record_test, self._process_test)

            if self.current_calibration_point >= self.num_calibration_points:
                print('--- Test Calibration Stage ---\n')
                self.logger.debug("Test Calibration complete.")

                print('Number of clusters created:', self.pog.get_num_clusters())
                self.print_aggregated_errors()
                
                self.calibration_completed[2] = True

        return self.fullscreen_frame

    def is_fully_calibrated(self):
        """
        Check if all calibration stages are complete.
        """
        return all(self.calibration_completed)

    # === Common Handler for a Calibration Stage Phase ===
    def _handle_stage_phase(self, data_collection_func, post_process_func):
        """
        A common handler that performs the three phases (instruction, fixation, data collection)
        for the given calibration stage.
        
        Parameters:
          data_collection_func: a function (with no parameters) that records data for the current frame.
          post_process_func: an optional function to process the collected data once data collection is complete.
        """

        # Phase 1: Instruction phase.
        if self.instruction_frame_count < self.instruction_frames:
            self._display_instruction()
            self.instruction_frame_count += 1
            return

        # Phase 2: Fixation phase.
        if self.fixation_frame_count < self.fixation_frames:
            self._display_fixation(self.current_calibration_point)
            self.fixation_frame_count += 1
            return

        # Phase 3: Data collection phase.
        if self.calibration_frame_count < self.calibration_frames:
            self._display_fixation(self.current_calibration_point)
            data_collection_func()  # stage-specific data recording.
            self.calibration_frame_count += 1
            return

        # Once enough frames have been collected for the current calibration point:
        post_process_func()  # stage-specific post-processing.
        
        # Then reset counters for this point and advance.
        self.current_calibration_point += 1
        self.fixation_frame_count = 0
        self.calibration_frame_count = 0
        self.fl = self.current_calibration_point

        return


    ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ###
    # --- Stage-Specific Callback Functions --- #

    # For the "ratios" stage:
    def _record_ratios(self):
        hr = self.gaze_tracking.horizontal_ratio()
        vr = self.gaze_tracking.vertical_ratio()
        
        if hr is not None and vr is not None:
            self.calibration_ratios[self.current_calibration_point].append([hr, vr])

        # check if the current calibration point is the center point.
        center_point = [self.frame_width // 2, self.frame_height // 2]
        if (self.calibration_points[self.current_calibration_point] == center_point).all():
            iris_diameter = self._measure_iris_diameter()
            if iris_diameter:
                self._update_iris_size(iris_diameter)

        if self.fl == self.current_calibration_point:
            print('\ncalib p', self.current_calibration_point)
            print('hr', hr)
            print('vr', vr)
            self.fl = -1

    def _process_ratios(self):
        ratios = self.calibration_ratios[self.current_calibration_point]
        best_ratios = GazeCalibration.cluster_calibration_ratios(ratios)
        target = self.calibration_points[self.current_calibration_point]
        # Save the representative ratios together with target coordinates.
        self.calibration_data[self.current_calibration_point] = (best_ratios[0], best_ratios[1], target[0], target[1])

    # For the "distortion" stage:
    def _record_distortion(self):
        raw = self.pog.point_of_gaze()  # using current mapping
        self._display_estimated_gaze(raw)
        if raw is not None:
            self.distortion_data[self.current_calibration_point].append(raw)
            # Optionally record error at each frame:
            self._record_error(raw)

    def _process_distortion(self):
        if self.distortion_data[self.current_calibration_point]:
            avg_x = np.median([pt[0] for pt in self.distortion_data[self.current_calibration_point]])
            avg_y = np.median([pt[1] for pt in self.distortion_data[self.current_calibration_point]])
            target = self.calibration_points[self.current_calibration_point]
            # Store as a tuple: (avg_raw_x, avg_raw_y, target_x, target_y)
            self.distortion_data[self.current_calibration_point] = (avg_x, avg_y, target[0], target[1])
            
    # For the "test" stage:
    def _record_test(self):
        raw = self.pog.point_of_gaze()
        corrected = self._undistort_feature(*raw) if raw is not None else (None, None)
        self._display_estimated_gaze(corrected)
        self._record_error(corrected)

    def _process_test(self):
        # For test stage, no additional processing is required; data is recorded per frame.
        pass



    ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ###
    # --- Calibration Data Processing Functions --- #

    def compute_polynomial_mapping(self):
        # Assume calibration_data is a list of tuples [(hr, vr, x, y), ...]
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

    def _update_iris_size(self, iris_d):
        self.base_iris_size += iris_d
        self.iris_size_count += 1

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



    ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ###
    # --- Distortion Calibration Functions --- #

    @staticmethod
    def distortion_model(params, x_norm, y_norm):
        """
        Given distortion parameters and normalized coordinates, compute the undistorted coordinates.
        params: array of 9 parameters [k1, k2, k3, p1, p2, s1, s2, s3, s4]
        where:
        k1, k2, k3: radial distortion coefficients,
        p1, p2: tangential distortion coefficients,
        s1, s2, s3, s4: prism distortion coefficients.
        """
        k1, k2, k3, p1, p2, s1, s2, s3, s4 = params
        p1, p2, k2, k3 = 0, 0, 0, 0
        
        # Compute the radial distance
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # Radial distortion factor (ρ)
        rho = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
        
        # Tangential distortion components (τ)
        tau_x = 2 * p1 * x_norm * y_norm + p2 * (r**2 + 2 * x_norm**2)
        tau_y = p1 * (r**2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
        
        # Prism distortion components (φ)
        phi_x = s1 * r**2 + s2 * r**4
        phi_y = s3 * r**2 + s4 * r**4
        
        # Apply the distortion model: corrected normalized coordinates
        x_corr = x_norm * rho + tau_x + phi_x
        y_corr = y_norm * rho + tau_y + phi_y
        
        return x_corr, y_corr

    @staticmethod
    def residuals(params, x_norm, y_norm, x_target, y_target):
        """
        Compute the residuals between the model's corrected normalized coordinates and the target normalized coordinates.
        """

        x_corr, y_corr = GazeCalibration.distortion_model(params, x_norm, y_norm)
        
        # Concatenate residuals for x and y into a single vector
        return np.concatenate([(x_corr - x_target), (y_corr - y_target)])

    def compute_distortion_coefficients(self):
        """
        Compute distortion coefficients from calibration data.
        
        Parameters:
            calibration_data: list of tuples [(x_raw, y_raw, x_target, y_target), ...]
                - x_raw, y_raw: raw coordinates from your polynomial mapping (before undistortion)
                - x_target, y_target: the actual screen coordinates for that calibration point.
            frame_width: screen width in pixels.
            frame_height: screen height in pixels.
        
        The function normalizes both raw and target coordinates to [-1, 1] using the screen center.
        
        Returns:
            params: the estimated distortion coefficients as a numpy array:
                    [k1, k2, k3, p1, p2, s1, s2, s3, s4]
        """

        data = np.array(self.distortion_data)
        x_raw = data[:, 0]
        y_raw = data[:, 1]
        x_target = data[:, 2]
        y_target = data[:, 3]
        
        # Normalize: using screen center and half dimensions to map to [-1, 1]
        x_norm = (x_raw - self.frame_width / 2) / (self.frame_width / 2)
        y_norm = (y_raw - self.frame_height / 2) / (self.frame_height / 2)
        x_target_norm = (x_target - self.frame_width / 2) / (self.frame_width / 2)
        y_target_norm = (y_target - self.frame_height / 2) / (self.frame_height / 2)
        
        # Initial guess for parameters
        initial_guess = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Run least squares optimization to find the distortion parameters
        result = least_squares(
            GazeCalibration.residuals, 
            initial_guess, 
            args=(x_norm, y_norm, x_target_norm, y_target_norm)
        )

        return result.x
    
    def _undistort_feature(self, x, y):
        """
        Correct the raw gaze coordinates (x,y) using the computed distortion coefficients.
        The coordinates are first normalized to [-1, 1] (with the screen center as 0,0),
        then passed through the distortion_model, and finally mapped back to screen coordinates.
        """
        # Convert raw coordinates to normalized coordinates.
        x_norm = (x - self.frame_width / 2) / (self.frame_width / 2)
        y_norm = (y - self.frame_height / 2) / (self.frame_height / 2)
        
        # Apply the distortion model using the computed coefficients.
        # Ensure self.distortion_params is available.
        if self.distortion_params is None:
            # If not available, return raw coordinates.
            return x, y
        x_corr_norm, y_corr_norm = GazeCalibration.distortion_model(self.distortion_params, x_norm, y_norm)
        
        # Map normalized coordinates back to screen coordinates.
        x_corr = x_corr_norm * (self.frame_width / 2) + self.frame_width / 2
        y_corr = y_corr_norm * (self.frame_height / 2) + self.frame_height / 2
        return int(x_corr), int(y_corr)



    ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### 
    # --- Public Helper Functions (display, record, errors) --- #

    def _display_instruction(self):
        """
        Display instructions on the fullscreen frame.
        """
        
        if self.calibration_stage == 'ratios':
            text = 'Ratios Calibration Stage'
        elif self.calibration_stage == 'distortion':
            text = 'Distortion Calibration Stage'
        elif self.calibration_stage == 'test':
            text = 'Test Calibration Stage'

        cv2.putText(
            self.fullscreen_frame,
            text,
            (80, 200),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (COLOR['text']),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            self.fullscreen_frame,
            'focus on the red dots',
            (80, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            (COLOR['text']),
            2,
            cv2.LINE_AA
        )

    def _display_fixation(self, calibration_index):
        """
        Display a fixation point (red circle) at the current calibration point.
        """
        cv2.circle(
            self.fullscreen_frame,
            self.calibration_points[calibration_index],
            self.circle_radius,
            (0, 0, 255),
            -1
        )

    def _display_estimated_gaze(self, raw):
        """
        Obtain the estimated gaze from the PointOfGaze object and display it.

        The gaze is shown as a small light-gray dot.
        """
        est_x, est_y = raw

        if est_x is not None and est_y is not None:
            cv2.circle(
                self.fullscreen_frame,
                (est_x, est_y),
                self.circle_radius // 4,
                (170, 170, 170),
                -1
            )

    def _calculate_error(self, screen_x, screen_y):
        """
        calculate and log the error between the estimated gaze coordinates and a target point.
        
        :param screen_x: estimated gaze x coordinate.
        :param screen_y: estimated gaze y coordinate.
        :return: tuple (err_x, err_y, err_xy) representing the error in x, error in y, and Euclidean error.
        """
        target_point = self.calibration_points[self.test_point_index]
        target_point = np.asarray(target_point)
        
        err_x = screen_x - target_point[0]
        err_y = screen_y - target_point[1]
        err_xy = np.linalg.norm(np.array([screen_x, screen_y]) - target_point)
        
        self.errors_dict['x'].append(err_x)
        self.errors_dict['y'].append(err_y)
        self.errors_dict['xy'].append(err_xy)
                    
    def _record_error(self, raw):
        """
        display the estimated gaze and record the error relative to a target point.
        
        the function draws a small light-gray dot at the estimated gaze and then calls
        calculate_error() to compute and record the error.
        """
        est_x, est_y = raw
        if est_x is None or est_y is None:
            return  # nothing to display or record
        
        # record the error using the common calculation function
        # this function will use self.calibration_points (or self.calibration_points) based on mode
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
