import cv2 # Keep for potential type hints or future internal use, but not drawing
import logging
import numpy as np

class GazeCalibration:
    """
    Class for calibrating gaze mapping to screen coordinates via a step-by-step process.

    This class manages the state of the calibration process (stages, points, timings)
    and computes the mapping parameters. The caller is responsible for driving the
    process frame-by-frame and handling the display based on the returned status.
    """

    # Calibration parameters (can be adjusted)
    DEFAULT_PARAMS = {
        "circle_radius": 20,
        "instruction_frames": 40, # Duration for showing stage intro
        "fixation_frames": 20,    # Duration for fixating before collection
        "calibration_frames": 40, # Duration for collecting data per point (ratios stage)
        "run_test_stage": True,   # Default to running the test stage
        "test_frames": 40         # Duration for testing per point (test stage)
    }

    # --- Initialization and Configuration --- #

    def __init__(self, gaze_tracking, monitor):
        """
        Initializes the GazeCalibration object.

        Args:
            gaze_tracking: The GazeTracking instance for accessing eye data.
            monitor (dict): Dictionary with screen dimensions {'width': W, 'height': H}.
        """
        self.logger = logging.getLogger(__name__)
        self.gaze_tracking = gaze_tracking
        if not isinstance(monitor, dict) or 'width' not in monitor or 'height' not in monitor:
             raise ValueError("GazeCalibration requires a valid monitor dictionary {'width': W, 'height': H}")
        self.monitor = monitor
        self.screen_width = monitor['width']
        self.screen_height = monitor['height']

        # Calibration results (polynomial coefficients)
        self.poly_x = None
        self.poly_y = None

        # Data storage
        self.errors_dict = {'x': [], 'y': [], 'xy': []}
        self.calibration_ratios_raw = [] # Store raw ratios per point temporarily
        self.calibration_data = [] # Store processed (clustered ratio, target) pairs

        self.test_stage_gaze_data = []
        # List of tuples: (frame_index_in_test_stage, point_index, landmarks_list)
        # landmarks_list itself is complex (list of NormalizedLandmark objects) - might need conversion
        self.test_stage_landmark_data = []
        self._test_stage_frame_counter = 0 # Track frames globally within test stage

        self.circle_radius = self.DEFAULT_PARAMS["circle_radius"]
        self.instruction_frames = self.DEFAULT_PARAMS["instruction_frames"]
        self.fixation_frames = self.DEFAULT_PARAMS["fixation_frames"]
        self.calibration_frames = self.DEFAULT_PARAMS["calibration_frames"]
        self.test_frames = self.DEFAULT_PARAMS["test_frames"]
        self.run_test_stage = self.DEFAULT_PARAMS["run_test_stage"] # Initialize flag

        # Internal state - managed by reset_calibration() and process_frame()
        self.current_stage = None # e.g., "ratios", "test"
        self.current_phase = None # e.g., "instruction", "fixation", "collecting"
        self.calibration_points = None
        self.num_calibration_points = 0
        self.current_point_index = 0
        self.frame_counter = 0 # Counter for frames within the current phase
        self._is_finished = False

        # Initialize state
        self.reset_calibration()
        self.logger.info(f"GazeCalibration initialized with default parameters: {self.get_current_parameters()}")

    def reset_calibration(self):
        """Resets the calibration state to start from the beginning."""
        self.logger.info("Resetting calibration state.")
        self._is_finished = False
        self.current_stage = "ratios"
        self.current_phase = "instruction"
        self.current_point_index = 0
        self.frame_counter = 0

        self.poly_x = None
        self.poly_y = None
        self.errors_dict = {'x': [], 'y': [], 'xy': []}

        # Generate points for the initial 'ratios' stage
        self._init_calibration_points(stage=self.current_stage)
        # Initialize data storage for the correct number of points
        self.calibration_ratios_raw = [[] for _ in range(self.num_calibration_points)]
        self.calibration_data = [] # Will be populated after 'ratios' stage

        # --- Clear test stage data stores ---
        self.test_stage_gaze_data = []
        self.test_stage_landmark_data = []
        self._test_stage_frame_counter = 0

        self.logger.info(f"Calibration reset. Starting stage: {self.current_stage}, phase: {self.current_phase}")

    def set_parameters(self, params):
         """
         Updates calibration parameters with proper type validation.

         Args:
             params (dict): Dictionary of parameters to update.
         """
         updated = False
         if not isinstance(params, dict):
              self.logger.error("set_parameters requires a dictionary.")
              return False # Indicate failure

         for key, value in params.items():
              if not hasattr(self, key):
                   self.logger.warning(f"Unknown calibration parameter '{key}'. Skipping.")
                   continue

              current_value = getattr(self, key)
              valid = False # Flag to check if value is valid for the key

              # --- Type and Value Validation ---
              if key == 'run_test_stage':
                   if isinstance(value, bool):
                        valid = True
                   else:
                        self.logger.warning(f"Invalid type for '{key}': {type(value)}. Expected bool. Skipping.")
              # Check other keys expected to be numbers (int or float)
              elif key in ['circle_radius', 'instruction_frames', 'fixation_frames', 'calibration_frames', 'test_frames']:
                   if isinstance(value, (int, float)):
                        if value > 0: # Ensure numeric params are positive
                             valid = True
                        else:
                             self.logger.warning(f"Invalid value for '{key}': {value}. Must be positive. Skipping.")
                   else:
                        self.logger.warning(f"Invalid type for '{key}': {type(value)}. Expected number. Skipping.")
              else:
                   # If we add more parameters later of different types, handle here
                   self.logger.warning(f"Validation not implemented for parameter '{key}'. Assuming invalid.")
                   valid = False # Default to invalid if type not handled
              # --- End Validation ---

              if valid and current_value != value:
                   setattr(self, key, value)
                   self.logger.info(f"Calibration parameter '{key}' updated from {current_value} to {value}")
                   updated = True
              elif valid and current_value == value:
                   # Value is valid but same as current, no update needed
                   pass

         return updated # Return True if any parameter was actually changed

    def get_current_parameters(self):
         """Returns the current values of the configurable parameters."""
         return {
              "circle_radius": self.circle_radius,
              "instruction_frames": self.instruction_frames,
              "fixation_frames": self.fixation_frames,
              "calibration_frames": self.calibration_frames,
              "test_frames": self.test_frames,
              "run_test_stage": self.run_test_stage,
         }

    def _init_calibration_points(self, stage):
        """Initializes calibration points based on the stage."""
        if stage == "ratios":
            # Use extreme points for initial ratio mapping
            self.calibration_points = self._generate_calibration_points(extreme_points=True)
            self.logger.debug(f"Initialized {len(self.calibration_points)} points for 'ratios' stage.")
        elif stage == "test":
            # Use inner points for testing the mapping
            self.calibration_points = self._generate_calibration_points(extreme_points=False)
            self.logger.debug(f"Initialized {len(self.calibration_points)} points for 'test' stage.")
        else:
             raise ValueError(f"Unknown calibration stage: {stage}")
        self.num_calibration_points = len(self.calibration_points)


    def _generate_calibration_points(self, rows=3, cols=3, margin_ratio=0.1, extreme_points=False):
        """Generates calibration points grid."""
        # Use self.screen_width and self.screen_height directly
        width = self.screen_width
        height = self.screen_height

        if extreme_points:
            margin = self.circle_radius # Ensure points are not exactly at 0,0 etc.
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



    # --- Main Step Processing Method ---

    def process_frame(self, point_of_gaze_estimator):
        """
        Processes a single frame/step in the calibration sequence.
        """
        if self._is_finished:
            return self._get_status_update(status='finished_all', display_type='message', text="Calibration Complete")

        self.frame_counter += 1
        status_update = None

        # --- Phase Logic ---
        if self.current_phase == "instruction":
            status_update = self._handle_instruction_phase()
        elif self.current_phase == "fixation":
            status_update = self._handle_fixation_phase()
        elif self.current_phase == "collecting":
            status_update = self._handle_collection_phase(point_of_gaze_estimator)
        else:
            # Should not happen
            self.logger.error(f"Invalid current phase: {self.current_phase}")
            return self._get_status_update(status='error', display_type='message', text="Internal calibration error: Invalid phase")

        # --- Phase/Point/Stage Transitions ---
        returned_status = status_update.get('status')

        if returned_status == 'finished_point':
            # Point finished, advance index
            self._advance_point()

            if self.current_point_index >= self.num_calibration_points:
                # Finished all points in this stage, trigger stage advancement
                return self._advance_stage()
            else:
                # Move to the *next* point, starting DIRECTLY with fixation
                self.current_phase = "fixation"
                self.frame_counter = 0
                target_point = self.calibration_points[self.current_point_index]
                return self._get_status_update(status='calibrating', phase='fixation',
                                               display_type='fixation_dot', target_point=target_point)

        elif returned_status == 'finished_stage':
             # This happens when _advance_stage computes mapping and transitions
             # The status returned by _advance_stage is the final one for this frame
             return status_update # Already contains the status for the start of the next stage

        elif returned_status == 'error' or returned_status == 'finished_all' or returned_status == 'calculating':
             # Let these statuses pass through directly
             return status_update

        elif returned_status == 'calibrating':
             # Continue in the current phase as indicated by the status_update
             return status_update
        else:
             # Catch unexpected status values
             self.logger.error(f"Unexpected status returned from phase handler: {returned_status}")
             return self._get_status_update(status='error', display_type='message', text="Internal calibration error: Unexpected status")


    # --- Phase Handlers ---

    def _handle_instruction_phase(self):
        """Handles the instruction phase, ONLY runs for the first point of a stage."""
        # This phase should only be active for current_point_index == 0
        if self.current_point_index != 0:
             self.logger.warning(f"Instruction phase entered for point index {self.current_point_index}. Should be skipped. Forcing fixation.")
             # Force transition to fixation immediately if entered incorrectly
             self.current_phase = "fixation"
             self.frame_counter = 0
             target_point = self.calibration_points[self.current_point_index]
             return self._get_status_update(status='calibrating', phase='fixation',
                                            display_type='fixation_dot', target_point=target_point)


        phase_duration = self.instruction_frames
        progress = self.frame_counter / phase_duration

        if self.frame_counter >= phase_duration:
            # Finished showing instruction, transition to fixation for the *first* point (index 0)
            self.current_phase = "fixation"
            self.frame_counter = 0
            target_point = self.calibration_points[0] # Explicitly use index 0
            return self._get_status_update(status='calibrating', phase='fixation',
                                           display_type='fixation_dot', target_point=target_point)
        else:
            # Continue showing the stage introduction text
            if self.current_stage == 'ratios':
                 text = 'Calibration: Ratios Stage\n\nPrepare to look at the dots.'
            elif self.current_stage == 'test':
                 text = 'Calibration: Test Stage\n\nPrepare to look at the dots.'
            else: text = 'Calibration Starting...' # Fallback

            return self._get_status_update(status='calibrating', phase='instruction',
                                           display_type='instruction_text', text=text,
                                           phase_progress=progress)


    def _handle_fixation_phase(self):
        """Handle logic and status updates for the fixation phase."""
        phase_duration = self.fixation_frames
        progress = self.frame_counter / phase_duration
        # Ensure target point exists for the current index
        if self.current_point_index >= self.num_calibration_points:
             self.logger.error(f"Fixation phase: Invalid point index {self.current_point_index}")
             return self._get_status_update(status='error', display_type='message', text="Internal calibration error: Invalid point index")
        target_point = self.calibration_points[self.current_point_index]


        if self.frame_counter >= phase_duration:
            # Transition to collection phase
            self.current_phase = "collecting"
            self.frame_counter = 0
            display_type = 'fixation_dot' if self.current_stage == 'ratios' else 'test_dot'
            return self._get_status_update(status='calibrating', phase='collecting',
                                           display_type=display_type, target_point=target_point)
        else:
            # Continue fixation phase
            display_type = 'fixation_dot' # Always show fixation dot during this phase
            return self._get_status_update(status='calibrating', phase='fixation',
                                           display_type=display_type, target_point=target_point,
                                           phase_progress=progress)


    def _handle_collection_phase(self, point_of_gaze_estimator):
        """Handle logic and status updates for the data collection phase."""
        phase_duration = self.calibration_frames if self.current_stage == 'ratios' else self.test_frames
        progress = self.frame_counter / phase_duration if phase_duration > 0 else 1.0

        if self.current_point_index >= self.num_calibration_points:
             self.logger.error(f"Collection phase: Invalid point index {self.current_point_index}")
             return self._get_status_update(status='error', display_type='message', text="Internal error: Invalid point index")
        target_point = self.calibration_points[self.current_point_index]

        estimated_gaze = None
        landmarks = None # Get landmarks for saving

        # --- Data Collection Logic ---
        if self.current_stage == 'ratios':
            self._collect_ratios_step()
            display_type = 'fixation_dot'
        elif self.current_stage == 'test':
            # Increment global test stage frame counter *before* collection for this frame
            self._test_stage_frame_counter += 1
            # Collect gaze, errors, AND landmarks
            estimated_gaze, landmarks = self._collect_test_step(point_of_gaze_estimator)
            display_type = 'test_dot'
        else:
             return self._get_status_update(status='error', display_type='message', text="Unknown collection stage")
        # --- End Data Collection ---


        # --- Check for Phase Completion ---
        if self.frame_counter >= phase_duration:
            # ... (post-collection processing for ratios/test point) ...
            if self.current_stage == 'ratios': self._process_ratios_for_point()
            elif self.current_stage == 'test': self._process_test_for_point()

            return self._get_status_update(status='finished_point', phase='collecting')
        else:
            # Continue collection phase
            return self._get_status_update(status='calibrating', phase='collecting',
                                           display_type=display_type, target_point=target_point,
                                           estimated_gaze=estimated_gaze,
                                           phase_progress=progress)


    # --- State Advancement ---

    def _advance_point(self):
        """ONLY increments the point index."""
        self.current_point_index += 1
        # Phase transition is handled in the main process_frame loop

    def _advance_stage(self):
        """Processes end-of-stage actions and transitions."""
        self.logger.info(f"Finished stage: {self.current_stage}")

        if self.current_stage == "ratios":
            # --- Compute Mapping (Always done after ratios stage) ---
            success = self.compute_polynomial_mapping()
            if not success:
                 self._is_finished = True
                 return self._get_status_update(status='error', display_type='message', text="Failed to compute mapping.")

            # --- Check if Test Stage should run ---
            if self.run_test_stage:
                # --- Transition to Test Stage ---
                self.logger.info("Proceeding to 'test' stage as requested.")
                self.current_stage = "test"
                self.current_point_index = 0
                self.current_phase = "instruction"
                self.frame_counter = 0
                self._init_calibration_points(stage=self.current_stage)
                # Reset test stage specific data stores
                self.errors_dict = {'x': [], 'y': [], 'xy': []}
                self.test_stage_gaze_data = []
                self.test_stage_landmark_data = []
                self._test_stage_frame_counter = 0
                # Return the status for the start of the test stage instruction phase
                return self._handle_instruction_phase()
            else:
                # --- Skip Test Stage and Finish ---
                self.logger.info("Skipping 'test' stage as requested.")
                self._is_finished = True
                # Return 'finished_all' status, indicating success but no test errors
                final_status = self._get_status_update(
                    status='finished_all',
                    display_type='message',
                    text="Calibration Complete (Test Stage Skipped)"
                )
                final_status['final_errors'] = None # Explicitly set errors to None
                final_status['test_data_summary'] = {'gaze_samples': 0, 'landmark_samples': 0}
                return final_status

        elif self.current_stage == "test":
            # --- Finalize Test Stage (if it ran) ---
            self.logger.info("Test stage complete.")
            aggregated_errors = self.get_aggregated_errors()
            self.print_aggregated_errors() # Keep for console logging

            self._is_finished = True
            # Return 'finished_all' status including test results
            final_status = self._get_status_update(status='finished_all', display_type='message',
                                                  text="Calibration Complete",
                                                  final_errors=aggregated_errors)
            final_status['test_data_summary'] = {
                 'gaze_samples': len(self.test_stage_gaze_data),
                 'landmark_samples': len(self.test_stage_landmark_data)
            }
            return final_status
        else:
             self._is_finished = True
             return self._get_status_update(status='error', display_type='message', text="Unknown stage finished")        


    # --- Helper: Build Status Dictionary ---

    def _get_status_update(self, status, phase=None, display_type=None, text=None, target_point=None, estimated_gaze=None, phase_progress=0.0, final_errors=None):
        """Constructs the standard status dictionary."""
        if phase is None: phase = self.current_phase # Default to current phase

        update = {
            'status': status, # 'calibrating', 'calculating', 'finished_point', 'finished_stage', 'finished_all', 'error'
            'stage': self.current_stage,
            'phase': phase,
            'progress': {
                'current_point': self.current_point_index,
                'total_points': self.num_calibration_points,
                'phase_progress': round(min(phase_progress, 1.0), 2) # Ensure progress is 0-1
            },
            'display_info': {
                'type': display_type, # 'instruction_text', 'fixation_dot', 'test_dot', 'message'
                # Convert numpy array to tuple for easier JSON serialization if needed later
                'target_point': tuple(target_point.astype(int)) if target_point is not None else None,
                'estimated_gaze': tuple(np.array(estimated_gaze).astype(int)) if estimated_gaze is not None else None,
                'text': text,
            }
        }
        if final_errors:
             update['final_errors'] = final_errors # Add errors if calibration finished

        return update


    # --- Data Collection/Processing Steps (Per Frame) ---

    def _collect_ratios_step(self):
        """Collects gaze ratios for the current frame during the 'ratios' stage."""
        hr = self.gaze_tracking.horizontal_ratio()
        vr = self.gaze_tracking.vertical_ratio()

        if hr is not None and vr is not None:
            # Append raw ratio for current point
             try:
                  self.calibration_ratios_raw[self.current_point_index].append([hr, vr])
             except IndexError:
                  self.logger.error(f"IndexError accessing calibration_ratios_raw at index {self.current_point_index}. Points: {self.num_calibration_points}")
                  # Handle error - maybe stop calibration? For now, just log.


    def _collect_test_step(self, point_of_gaze_estimator):
        """Collects estimated gaze, errors, and landmarks during 'test' stage."""
        est_x, est_y = point_of_gaze_estimator.point_of_gaze()
        # Access landmarks directly from the gaze_tracking object (refreshed before process_frame)
        landmarks_mp = self.gaze_tracking.landmarks # This is the list of NormalizedLandmark objects

        estimated_gaze_tuple = None
        if est_x is not None and est_y is not None:
            target = self.calibration_points[self.current_point_index]
            err_x = est_x - target[0]
            err_y = est_y - target[1]
            err_xy = np.linalg.norm(np.array([est_x, est_y]) - target)

            # Store frame-by-frame errors
            self.errors_dict['x'].append(err_x)
            self.errors_dict['y'].append(err_y)
            self.errors_dict['xy'].append(err_xy)

            # --- Store detailed gaze data ---
            self.test_stage_gaze_data.append((
                self._test_stage_frame_counter,
                self.current_point_index,
                target[0], target[1],
                est_x, est_y
            ))
            estimated_gaze_tuple = (est_x, est_y)
            # --- End Store gaze ---

        # --- Store landmark data (even if gaze is None, landmarks might exist) ---
        if landmarks_mp:
            # Convert landmarks to a savable format (e.g., list of [x, y, z]) immediately
            landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks_mp]
            self.test_stage_landmark_data.append((
                self._test_stage_frame_counter,
                self.current_point_index,
                landmarks_list # Store the converted list
            ))
        # --- End Store landmarks ---

        # Return gaze for display, and landmarks for internal use/logging if needed
        return estimated_gaze_tuple, landmarks_mp # Return original MP landmarks here

    # --- Data Processing Steps (End of Point/Stage) ---

    def _process_ratios_for_point(self):
        """Processes the collected raw ratios for the finished calibration point."""
        try:
            ratios = self.calibration_ratios_raw[self.current_point_index]
            if not ratios:
                self.logger.warning(f"No ratios collected for calibration point {self.current_point_index}.")
                # How to handle this? Skip point? Abort? For now, log and continue.
                # We might need a minimum number of valid samples per point.
                # Add a placeholder ratio/target pair? Risky. Best to skip.
                return # Skip adding data for this point

            best_ratios = GazeCalibration.cluster_calibration_ratios(ratios)
            if best_ratios is None:
                 self.logger.warning(f"Could not cluster ratios for point {self.current_point_index}.")
                 return # Skip adding data

            target = self.calibration_points[self.current_point_index]
            # Store the representative ratio pair and target coordinates
            self.calibration_data.append((best_ratios[0], best_ratios[1], target[0], target[1]))
            self.logger.debug(f"Processed point {self.current_point_index}: Ratios ({best_ratios[0]:.3f}, {best_ratios[1]:.3f}), Target ({target[0]}, {target[1]})")
        except IndexError:
            self.logger.error(f"IndexError processing ratios for point {self.current_point_index}")

    def _process_test_for_point(self):
        """Placeholder for any processing needed at the end of a test point."""
        # Currently, errors are aggregated per frame in _collect_test_step.
        # We could calculate mean error for the point here if needed.
        self.logger.debug(f"Finished collecting test data for point {self.current_point_index}")
        pass

    def compute_polynomial_mapping(self):
        """Computes the polynomial mapping from collected ratio/target data."""
        if not self.calibration_data or len(self.calibration_data) < 6: # Need at least 6 points for 6 coefficients
            self.logger.error(f"Insufficient valid calibration data ({len(self.calibration_data)} points) to compute mapping.")
            self.poly_x = None
            self.poly_y = None
            return False

        data = np.array(self.calibration_data) # Shape: (num_points, 4)
        hr = data[:, 0]
        vr = data[:, 1]
        x_values = data[:, 2]
        y_values = data[:, 3]

        # Build design matrix A: [hr^2, vr^2, hr*vr, hr, vr, 1]
        A = np.column_stack([hr**2, vr**2, hr*vr, hr, vr, np.ones_like(hr)])

        try:
            # Solve least squares for x mapping and y mapping
            self.poly_x, res_x, _, _ = np.linalg.lstsq(A, x_values, rcond=None)
            self.poly_y, res_y, _, _ = np.linalg.lstsq(A, y_values, rcond=None)
            self.logger.info("Polynomial mapping computed successfully.")
            # Optionally log residuals if needed: self.logger.debug(f"Lstsq residuals: x={res_x}, y={res_y}")
            return True
        except np.linalg.LinAlgError as e:
            self.logger.error(f"Linear algebra error computing polynomial mapping: {e}")
            self.poly_x = None
            self.poly_y = None
            return False

    # --- Data Retrieval Methods ---

    def get_test_stage_gaze_data(self):
         """
         Returns the detailed gaze data collected during the test stage.

         Returns:
             list[tuple]: List of (frame_idx, point_idx, target_x, target_y, est_x, est_y)
         """
         return self.test_stage_gaze_data

    def get_test_stage_landmark_data(self):
         """
         Returns the landmark data collected during the test stage.

         Returns:
             list[tuple]: List of (frame_idx, point_idx, landmarks)
                          where landmarks is a list of [x, y, z] coordinates.
         """
         return self.test_stage_landmark_data


    # --- Utility methods ---

    def is_finished(self):
        """Check if the entire calibration process is complete."""
        return self._is_finished

    def get_polynomials(self):
         """Returns the computed polynomial coefficients."""
         if self.poly_x is not None and self.poly_y is not None:
              return {'poly_x': self.poly_x, 'poly_y': self.poly_y}
         else:
              return None

    @staticmethod
    def density_cluster_1d(data):
        if data is None or len(data) == 0: return None
        try:
             hist, bins = np.histogram(data, bins='auto')
             if len(hist) == 0: return np.mean(data) # Fallback if histogram fails
             idx = int(np.argmax(hist))
             return (bins[idx] + bins[idx + 1]) / 2
        except Exception as e:
             # Fallback if histogramming fails for any reason
             logging.getLogger(__name__).warning(f"Histogram clustering failed: {e}. Falling back to mean.")
             return np.mean(data)


    @staticmethod
    def cluster_calibration_ratios(ratios):
        if not ratios: return None
        ratios_array = np.array(ratios)
        if ratios_array.shape[1] != 2: return None # Expect pairs

        horizontal_values = ratios_array[:, 0]
        vertical_values = ratios_array[:, 1]

        best_horizontal = GazeCalibration.density_cluster_1d(horizontal_values)
        best_vertical = GazeCalibration.density_cluster_1d(vertical_values)

        if best_horizontal is None or best_vertical is None:
            return None

        return [best_horizontal, best_vertical]

    # --- Error Reporting ---

    def get_aggregated_errors(self):
        """Calculates and returns the mean absolute errors from the test stage."""
        if not self.errors_dict or not self.errors_dict['xy']: # Check if any errors were recorded
            return {'mean_x': None, 'mean_y': None, 'mean_xy': None, 'count': 0}

        count = len(self.errors_dict['xy'])
        mean_x = np.mean(np.abs(self.errors_dict['x'])) if self.errors_dict['x'] else None
        mean_y = np.mean(np.abs(self.errors_dict['y'])) if self.errors_dict['y'] else None
        mean_xy = np.mean(np.abs(self.errors_dict['xy'])) # xy should always exist if count > 0

        return {'mean_x': mean_x, 'mean_y': mean_y, 'mean_xy': mean_xy, 'count': count}

    def print_aggregated_errors(self):
        """Prints the aggregated estimation errors from the test stage."""
        errors = self.get_aggregated_errors()
        if errors['count'] > 0:
            print(f"\n--- Calibration Test Stage Errors ({errors['count']} samples) ---")
            print(f"Mean Absolute Error X : {errors['mean_x']:.2f} pixels")
            print(f"Mean Absolute Error Y : {errors['mean_y']:.2f} pixels")
            print(f"Mean Absolute Error XY: {errors['mean_xy']:.2f} pixels")
            print("--------------------------------------------------")
        else:
            print("\nNo error data collected during test stage.")
