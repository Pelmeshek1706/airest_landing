import logging
import numpy as np

class GazeCalibration:
    """
    Manages the state and logic for an interactive, user-driven gaze calibration process.
    The process is driven by the user holding and releasing the space bar, rather than
    fixed timers, to ensure focused data collection for each calibration point.
    """

    DEFAULT_PARAMS = {
        "circle_radius": 20,
        "outer_circle_initial_radius": 80,
        "instruction_frames": 40,
        "calibration_frames": 60,
        "run_test_stage": True,
        "test_frames": 40
    }

    PHASE_INSTRUCTION = "instruction"
    PHASE_AWAITING_TRIGGER = "awaiting_trigger"
    PHASE_COLLECTING = "collecting"
    PHASE_POINT_COMPLETED = "point_completed"


    def __init__(self, gaze_tracking, monitor):
        self.logger = logging.getLogger(__name__)
        self.gaze_tracking = gaze_tracking
        if not isinstance(monitor, dict) or 'width' not in monitor or 'height' not in monitor:
             raise ValueError("GazeCalibration requires a valid monitor dictionary")
        self.monitor = monitor
        self.screen_width = monitor['width']
        self.screen_height = monitor['height']

        self.set_parameters(self.DEFAULT_PARAMS)
        self.reset_calibration()
        self.logger.info(f"GazeCalibration initialized with parameters: {self.get_current_parameters()}")

    def reset_calibration(self):
        self.logger.info("Resetting calibration state.")
        self._is_finished = False
        self.current_stage = "ratios"
        self.current_phase = self.PHASE_INSTRUCTION
        self.current_point_index = 0
        self.frame_counter = 0

        self.poly_x, self.poly_y = None, None
        self.errors_dict = {'x': [], 'y': [], 'xy': []}
        
        self.calibration_data = []
        self.test_stage_gaze_data = []
        self.test_stage_landmark_data = []
        self._test_stage_frame_counter = 0
        
        self._init_calibration_points(stage=self.current_stage)

    def set_parameters(self, params):
        if not isinstance(params, dict):
            self.logger.error("set_parameters requires a dictionary.")
            return
        for key, value in params.items():
            setattr(self, key, value)

    def get_current_parameters(self):
         return {k: getattr(self, k, None) for k in self.DEFAULT_PARAMS}

    def _init_calibration_points(self, stage):
        self.calibration_points = self._generate_calibration_points(extreme_points=(stage == "ratios"))
        self.num_calibration_points = len(self.calibration_points)
        self.calibration_ratios_raw = [[] for _ in range(self.num_calibration_points)]
        self.logger.debug(f"Initialized {self.num_calibration_points} points for '{stage}' stage.")

    def _generate_calibration_points(self, rows=3, cols=3, margin_ratio=0.1, extreme_points=False):
        width, height = self.screen_width, self.screen_height
        if extreme_points:
            margin = self.circle_radius
            xs = np.linspace(margin, width - margin, cols, dtype=int)
            ys = np.linspace(margin, height - margin, rows, dtype=int)
        else:
            margin_x, margin_y = margin_ratio * width, margin_ratio * height
            xs = np.linspace(margin_x, width - margin_x, cols, dtype=int)
            ys = np.linspace(margin_y, height - margin_y, rows, dtype=int)
        
        return np.array([[x, y] for y in ys for x in xs])

    def get_initial_status(self):
        text = f"Calibration: {self.current_stage.capitalize()} Stage\n\nHold SPACE to calibrate each point.\nRelease to advance to the next."
        return self._get_status_update('calibrating', display_type='instruction_text', text=text)

    def process_frame(self, point_of_gaze_estimator, user_input=None):
        if self._is_finished:
            return self._get_status_update('finished_all', display_type='message', text="Calibration Complete")
        
        user_input = user_input or {}
        space_down = user_input.get('space_down', False)

        if self.current_phase == self.PHASE_INSTRUCTION:
            return self._handle_instruction_phase()
        
        if self.current_phase == self.PHASE_AWAITING_TRIGGER:
            return self._handle_awaiting_trigger_phase(space_down)

        if self.current_phase == self.PHASE_COLLECTING:
            return self._handle_collection_phase(point_of_gaze_estimator, space_down)
        
        if self.current_phase == self.PHASE_POINT_COMPLETED:
            return self._handle_point_completed_phase(space_down)

        return self._get_status_update('error', display_type='message', text='Internal Error: Unknown calibration phase')

    def _handle_instruction_phase(self):
        self.frame_counter += 1
        if self.frame_counter >= self.instruction_frames:
            self.current_phase = self.PHASE_AWAITING_TRIGGER
            self.frame_counter = 0
        return self.get_initial_status()

    def _handle_awaiting_trigger_phase(self, space_down):
        if space_down:
            self.current_phase = self.PHASE_COLLECTING
            self.frame_counter = 0
            self.logger.info(f"User triggered collection for point {self.current_point_index}")
        
        target_point = self.calibration_points[self.current_point_index]
        return self._get_status_update('calibrating', target_point=target_point)

    def _handle_collection_phase(self, pog_estimator, space_down):
        duration = self.test_frames if self.current_stage == 'test' else self.calibration_frames
        
        if not space_down:
            self.frame_counter = 0
            # bug fix: add boundary check before accessing the list index
            if self.current_point_index < self.num_calibration_points:
                self.calibration_ratios_raw[self.current_point_index] = []
            self.current_phase = self.PHASE_AWAITING_TRIGGER
            self.logger.warning(f"Collection for point {self.current_point_index} interrupted, data discarded.")
            target_point = self.calibration_points[self.current_point_index]
            return self._get_status_update('calibrating', target_point=target_point)

        self.frame_counter += 1
        estimated_gaze = None
        
        if self.current_stage == 'ratios':
            self._collect_ratios_step()
        elif self.current_stage == 'test':
            self._test_stage_frame_counter += 1
            estimated_gaze, _ = self._collect_test_step(pog_estimator)
        
        if self.frame_counter >= duration:
            if self.current_stage == 'ratios': self._process_ratios_for_point()
            self.current_phase = self.PHASE_POINT_COMPLETED
            self.logger.info(f"Collection finished for point {self.current_point_index}")
        
        target_point = self.calibration_points[self.current_point_index]
        return self._get_status_update('calibrating', target_point=target_point, estimated_gaze=estimated_gaze)

    def _handle_point_completed_phase(self, space_down):
        if not space_down:
            self.logger.info(f"User released trigger, advancing from point {self.current_point_index}")
            return self._advance_point()
        
        target_point = self.calibration_points[self.current_point_index]
        return self._get_status_update('calibrating', target_point=target_point)

    def _advance_point(self):
        self.current_point_index += 1
        if self.current_point_index >= self.num_calibration_points:
            return self._advance_stage()
        
        self.current_phase = self.PHASE_AWAITING_TRIGGER
        target_point = self.calibration_points[self.current_point_index]
        return self._get_status_update('calibrating', target_point=target_point)

    def _advance_stage(self):
        self.logger.info(f"Finished all points in stage: {self.current_stage}")

        if self.current_stage == "ratios":
            if not self.compute_polynomial_mapping():
                 self._is_finished = True
                 return self._get_status_update('error', display_type='message', text="Failed to compute gaze mapping.")

            if self.run_test_stage:
                self.logger.info("Proceeding to 'test' stage.")
                self.current_stage = "test"
                self.current_point_index = 0
                self.current_phase = self.PHASE_INSTRUCTION
                self.frame_counter = 0
                self._init_calibration_points(stage=self.current_stage)
                return self.get_initial_status()

        self._is_finished = True
        
        final_errors = self.get_aggregated_errors() if self.current_stage == "test" else None
        if final_errors and final_errors['count'] > 0:
            self.print_aggregated_errors()
        
        return self._get_status_update('finished_all', display_type='message', text="Calibration Complete", final_errors=final_errors)

    def _get_status_update(self, status, display_type=None, text=None, target_point=None, estimated_gaze=None, final_errors=None):
        phase = self.current_phase
        display_type = display_type or ('test_dot' if self.current_stage == 'test' else 'fixation_dot')

        outer_radius, is_completed = None, False
        if phase == self.PHASE_AWAITING_TRIGGER:
            outer_radius = self.outer_circle_initial_radius
        elif phase == self.PHASE_COLLECTING:
            duration = self.test_frames if self.current_stage == 'test' else self.calibration_frames
            progress = self.frame_counter / duration if duration > 0 else 1.0
            outer_radius = self.circle_radius + (self.outer_circle_initial_radius - self.circle_radius) * (1.0 - progress)
            outer_radius = max(self.circle_radius, outer_radius)
        elif phase == self.PHASE_POINT_COMPLETED:
            is_completed = True
            
        display_info = {
            'type': display_type, 'text': text,
            'target_point': target_point.tolist() if target_point is not None else None,
            'estimated_gaze': estimated_gaze,
            'outer_circle_radius': outer_radius,
            'is_point_completed': is_completed,
            'inner_circle_radius': self.circle_radius,
            'outer_circle_initial_radius': self.outer_circle_initial_radius,
        }

        update = {
            'status': status, 'stage': self.current_stage, 'phase': phase,
            'point_index': self.current_point_index, 'total_points': self.num_calibration_points,
            'display_info': display_info
        }
        
        if final_errors:
            update['final_errors'] = final_errors
            
        return update

    def compute_polynomial_mapping(self):
        if len(self.calibration_data) < 6:
            self.logger.error(f"Cannot compute mapping: only {len(self.calibration_data)} valid data points collected.")
            return False
        
        data = np.array(self.calibration_data)
        ratios_x, ratios_y = data[:, 0], data[:, 1]
        targets_x, targets_y = data[:, 2], data[:, 3]

        A = np.c_[ratios_x**2, ratios_y**2, ratios_x * ratios_y, ratios_x, ratios_y, np.ones_like(ratios_x)]
        
        try:
            self.poly_x = np.linalg.lstsq(A, targets_x, rcond=None)[0]
            self.poly_y = np.linalg.lstsq(A, targets_y, rcond=None)[0]
            self.logger.info("Successfully computed polynomial mapping.")
            return True
        except np.linalg.LinAlgError as e:
            self.logger.error(f"Failed to compute polynomial mapping: {e}")
            self.poly_x, self.poly_y = None, None
            return False

    def _collect_ratios_step(self):
        h_ratio = self.gaze_tracking.horizontal_ratio()
        v_ratio = self.gaze_tracking.vertical_ratio()
        if h_ratio is not None and v_ratio is not None:
            self.calibration_ratios_raw[self.current_point_index].append((h_ratio, v_ratio))

    def _process_ratios_for_point(self):
        ratios = self.calibration_ratios_raw[self.current_point_index]
        if not ratios: 
            self.logger.warning(f"No ratios collected for point {self.current_point_index}, skipping.")
            return
        
        avg_ratio = np.mean(ratios, axis=0)
        target = self.calibration_points[self.current_point_index]
        self.calibration_data.append(np.concatenate((avg_ratio, target)))

    def _collect_test_step(self, pog_estimator):
        if self.poly_x is None or self.poly_y is None: return None, None
        
        estimated_gaze = pog_estimator._compute_raw_gaze()
        landmarks = self.gaze_tracking.landmarks
        target_point = self.calibration_points[self.current_point_index]

        if estimated_gaze and estimated_gaze[0] is not None:
            gaze_x, gaze_y = estimated_gaze
            err_x = gaze_x - target_point[0]
            err_y = gaze_y - target_point[1]
            err_xy = np.linalg.norm(np.array(estimated_gaze) - target_point)
            
            self.errors_dict['x'].append(err_x)
            self.errors_dict['y'].append(err_y)
            self.errors_dict['xy'].append(err_xy)
            
            self.test_stage_gaze_data.append((self._test_stage_frame_counter, self.current_point_index, target_point[0], target_point[1], gaze_x, gaze_y))
        
        if landmarks:
            self.test_stage_landmark_data.append((self._test_stage_frame_counter, self.current_point_index, landmarks))

        return estimated_gaze, landmarks
    
    def get_aggregated_errors(self):
        if not self.errors_dict or not self.errors_dict['xy']:
            return {'mean_x': None, 'mean_y': None, 'mean_xy': None, 'count': 0}

        count = len(self.errors_dict['xy'])
        mean_x = float(np.mean(np.abs(self.errors_dict['x'])))
        mean_y = float(np.mean(np.abs(self.errors_dict['y'])))
        mean_xy = float(np.mean(np.abs(self.errors_dict['xy'])))

        return {'mean_x': mean_x, 'mean_y': mean_y, 'mean_xy': mean_xy, 'count': count}

    def print_aggregated_errors(self):
        errors = self.get_aggregated_errors()
        if errors['count'] > 0:
            self.logger.info(f"--- Calibration Test Stage Errors ({errors['count']} samples) ---")
            self.logger.info(f"Mean Absolute Error X : {errors['mean_x']:.2f} pixels")
            self.logger.info(f"Mean Absolute Error Y : {errors['mean_y']:.2f} pixels")
            self.logger.info(f"Mean Euclidian Error XY: {errors['mean_xy']:.2f} pixels")
            self.logger.info("--------------------------------------------------")
        else:
            self.logger.info("No error data was collected during the test stage.")

    def is_finished(self):
        return self._is_finished