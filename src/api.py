import cv2
import numpy as np
import logging
import os
import pandas as pd

from .tracking.gaze_tracking import GazeTracking
from .calibration.gaze_calibration import GazeCalibration
from .tracking.point_of_gaze import PointOfGaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GazeTrackerAPI:
    """
    API for initializing, calibrating, and using the gaze tracking system.

    This class provides a high-level interface to the gaze tracking subproject.
    The caller is responsible for providing camera frames and screen dimensions.
    It handles calibration state and real-time gaze/landmark estimation.
    """

    def __init__(self, screen_width, screen_height, stabilize_gaze=True, calibration_file=None):
        """
        Initializes the GazeTrackerAPI configuration.
        """
        logger.info("Initializing GazeTrackerAPI...")
        if not isinstance(screen_width, int) or not isinstance(screen_height, int) or screen_width <= 0 or screen_height <= 0:
            raise ValueError("Screen width and height must be positive integers.")

        self.stabilize = stabilize_gaze
        self.calibration_file_path = calibration_file
        self.monitor = {'width': screen_width, 'height': screen_height}

        self.gaze_tracking = None
        self.gaze_calibration = None
        self.pog = None
        self._is_calibrated = False
        self._is_running = False
        self._is_calibrating = False

        logger.info(f"Configuration: Screen={self.monitor['width']}x{self.monitor['height']}, Stabilize={self.stabilize}, CalibFile={self.calibration_file_path}")

    def start(self):
        """
        Initializes tracking components using the provided screen dimensions.
        """
        if self._is_running:
            logger.warning("API is already running. Call stop() first if you need to restart.")
            return

        logger.info("Starting GazeTrackerAPI resources...")
        if not self.monitor or 'width' not in self.monitor or 'height' not in self.monitor:
             raise RuntimeError("Monitor dimensions not set during initialization.")
        logger.info(f"Using Screen Size: {self.monitor['width']}x{self.monitor['height']}")

        try:
            logger.info("Initializing GazeTracking (MediaPipe)...")
            self.gaze_tracking = GazeTracking()

            logger.info("Initializing GazeCalibration...")
            self.gaze_calibration = GazeCalibration(self.gaze_tracking, self.monitor)
            
            logger.info("Initializing PointOfGaze...")
            self.pog = PointOfGaze(self.gaze_tracking, self.gaze_calibration, self.monitor, self.stabilize)

            if self.calibration_file_path and os.path.exists(self.calibration_file_path):
                self.load_calibration(self.calibration_file_path)

            self._is_running = True
            logger.info("GazeTrackerAPI started successfully.")

        except Exception as e:
            logger.error(f"Error during API start: {e}")
            self.stop()
            raise

    def load_calibration(self, filepath):
        """
        Loads calibration parameters from a file.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before loading calibration.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        if not self.gaze_calibration:
             raise RuntimeError("GazeCalibration object not initialized. Cannot load data.")

        logger.info(f"Loading calibration data from {filepath}...")
        try:
            data = np.load(filepath)
            if 'poly_x' not in data or 'poly_y' not in data:
                raise ValueError("Calibration file is missing required keys ('poly_x', 'poly_y').")

            if 'screen_width' in data and 'screen_height' in data:
                loaded_w, loaded_h = data['screen_width'], data['screen_height']
                current_w, current_h = self.monitor['width'], self.monitor['height']
                if int(loaded_w) != current_w or int(loaded_h) != current_h:
                    logger.warning(f"Loaded calibration screen size ({loaded_w}x{loaded_h}) "
                                   f"differs from current API screen size ({current_w}x{current_h}).")
            
            self.gaze_calibration.poly_x = data['poly_x']
            self.gaze_calibration.poly_y = data['poly_y']
            self._is_calibrated = True
            self.calibration_file_path = filepath
            logger.info("Calibration data loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration data from {filepath}: {e}")
            self._is_calibrated = False
            raise

    def save_calibration(self, filepath):
        """
        Saves the current calibration parameters and screen dimensions to a file.
        """
        if not self._is_calibrated:
            raise RuntimeError("System is not calibrated. Cannot save calibration data.")
        if not self.gaze_calibration or self.gaze_calibration.poly_x is None:
             raise RuntimeError("Calibration data (polynomials) not found.")
        if not self.monitor:
             raise RuntimeError("Monitor dimensions not available for saving.")

        logger.info(f"Saving calibration data to {filepath}...")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez(
                filepath,
                poly_x=self.gaze_calibration.poly_x,
                poly_y=self.gaze_calibration.poly_y,
                screen_width=self.monitor['width'],
                screen_height=self.monitor['height']
            )
            logger.info("Calibration data saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save calibration data to {filepath}: {e}")
            raise IOError(f"Failed to save calibration data: {e}")

    def stop(self):
        """
        Releases resources and cleans up internal state.
        """
        if not self._is_running:
            return

        logger.info("Stopping GazeTrackerAPI resources...")
        if self.gaze_tracking and hasattr(self.gaze_tracking.face_mesh, 'close'):
             try:
                 self.gaze_tracking.face_mesh.close()
                 logger.info("MediaPipe FaceMesh resources released.")
             except Exception as e:
                 logger.warning(f"Exception while closing MediaPipe FaceMesh: {e}")

        self.gaze_tracking = None
        self.gaze_calibration = None
        self.pog = None
        self._is_running = False
        self._is_calibrating = False
        logger.info("GazeTrackerAPI stopped.")

    def start_calibration(self, calibration_params=None):
        """
        Resets and starts the interactive calibration process.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before calibration.")
        if not self.gaze_calibration or not self.pog:
             raise RuntimeError("Calibration components not initialized.")

        if calibration_params is not None:
             if not isinstance(calibration_params, dict):
                  raise TypeError("calibration_params must be a dictionary or None.")
             self.gaze_calibration.set_parameters(calibration_params)
             logger.info(f"Applied custom calibration parameters: {calibration_params}")

        self.gaze_calibration.reset_calibration()
        self._is_calibrating = True
        self._is_calibrated = False
        logger.info("Calibration process started/restarted.")

        return self.gaze_calibration.get_initial_status()

    def process_calibration_step(self, frame, user_input=None):
        """
        Processes one frame/step of the calibration sequence.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() first.")
        if not self._is_calibrating:
            raise RuntimeError("Calibration not started. Call start_calibration() first.")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame provided to process_calibration_step.")

        self.gaze_tracking.refresh(frame)
        status_update = self.gaze_calibration.process_frame(self.pog, user_input=user_input)

        returned_status = status_update.get('status')
        if returned_status in ('finished_all', 'error'):
            self._is_calibrated = (returned_status == 'finished_all')
            self._is_calibrating = False
            log_level = logging.INFO if self._is_calibrated else logging.ERROR
            logger.log(log_level, f"Calibration process ended with status: {returned_status}")
            
        return status_update

    def cancel_calibration(self):
        """
        Stops the currently active calibration process.
        """
        if not self._is_calibrating:
            return

        logger.warning("Cancelling calibration process.")
        self._is_calibrating = False
        self._is_calibrated = False
        self.gaze_calibration.reset_calibration()
        logger.info("Calibration cancelled.")

    def get_gaze(self, frame):
        """
        Estimates the gaze position on the screen for a given camera frame.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before getting gaze.")
        if not self._is_calibrated:
            return (None, None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame provided to get_gaze.")

        self.gaze_tracking.refresh(frame)
        estimated_gaze = self.pog.point_of_gaze()
        return estimated_gaze
    
    def get_landmarks(self, frame):
        """
        Processes the frame and returns the detected MediaPipe face landmarks.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before getting landmarks.")
        if not self.gaze_tracking:
             logger.warning("GazeTracking not initialized. Cannot get landmarks.")
             return None
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame provided to get_landmarks.")

        self.gaze_tracking.refresh(frame)
        
        if self.gaze_tracking.landmarks:
            return self.gaze_tracking.landmarks
        return None

    def get_calibration_test_data(self):
        """
        Retrieves detailed gaze and landmark data collected during the calibration's test stage.
        """
        if not self.gaze_calibration:
            return [], []
        return self.gaze_calibration.test_stage_gaze_data, self.gaze_calibration.test_stage_landmark_data

    def save_calibration_test_data(self, gaze_filepath, landmarks_filepath):
        """
        Saves the calibration test stage data to specified CSV files.
        """
        if not self.gaze_calibration:
            logger.warning("GazeCalibration object not found, cannot save test data.")
            return False

        gaze_data, landmark_raw_data = self.get_calibration_test_data()
        gaze_saved, landmarks_saved = False, False

        if gaze_data:
            try:
                df_gaze = pd.DataFrame(gaze_data, columns=['frame_idx', 'point_idx', 'target_x', 'target_y', 'gaze_x', 'gaze_y'])
                df_gaze.to_csv(gaze_filepath, index=False)
                logger.info(f"Calibration test gaze data saved to {gaze_filepath}")
                gaze_saved = True
            except Exception as e:
                logger.error(f"Failed to save gaze test data: {e}")
        
        if landmark_raw_data:
            try:
                landmark_rows = []
                for frame_idx, point_idx, landmarks_list in landmark_raw_data:
                    for lm_idx, lm in enumerate(landmarks_list):
                        landmark_rows.append([frame_idx, point_idx, lm_idx, lm.x, lm.y, lm.z])
                df_landmarks = pd.DataFrame(landmark_rows, columns=['frame_idx', 'point_idx', 'landmark_id', 'x', 'y', 'z'])
                df_landmarks.to_csv(landmarks_filepath, index=False)
                logger.info(f"Calibration test landmark data saved to {landmarks_filepath}")
                landmarks_saved = True
            except Exception as e:
                logger.error(f"Failed to save landmark test data: {e}")

        return gaze_saved or landmarks_saved

    @property
    def is_calibrated(self):
        return self._is_calibrated

    @property
    def is_calibrating(self):
        return self._is_calibrating

    @property
    def is_running(self):
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()