import cv2 # Keep cv2 import for frame type hints maybe, but not for window/camera
import numpy as np
import logging
import os
import pandas as pd

# Import necessary components from your project structure
# REMOVED: from .utils.utils import get_screen_size
from .tracking.gaze_tracking import GazeTracking
from .calibration.gaze_calibration import GazeCalibration
from .tracking.point_of_gaze import PointOfGaze

# Basic logging configuration (optional, but good practice)
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

        Args:
            screen_width (int): The width of the screen or display area (in pixels)
                                where gaze is being projected.
            screen_height (int): The height of the screen or display area (in pixels)
                                 where gaze is being projected.
            stabilize_gaze (bool): Whether to enable gaze stabilization. Defaults to True.
            calibration_file (str, optional): Path to a file containing pre-saved
                                             calibration data. Defaults to None.

        Raises:
            ValueError: If screen_width or screen_height are not positive integers.
        """
        logger.info("Initializing GazeTrackerAPI...")
        if not isinstance(screen_width, int) or not isinstance(screen_height, int) or screen_width <= 0 or screen_height <= 0:
            raise ValueError("Screen width and height must be positive integers.")

        self.stabilize = stabilize_gaze
        self.calibration_file_path = calibration_file

        # Store screen dimensions provided by the caller
        self.monitor = {'width': screen_width, 'height': screen_height}

        # Internal state variables - initialized in start() or load_calibration()
        self.gaze_tracking = None
        self.gaze_calibration = None
        self.pog = None # PointOfGaze instance
        self._is_calibrated = False
        self._is_running = False # Flag to track if start() has been called

        # Calibration process state (will be managed by calibration methods)
        self._is_calibrating = False

        logger.info(f"Configuration: Screen={self.monitor['width']}x{self.monitor['height']}, Stabilize={self.stabilize}, CalibFile={self.calibration_file_path}")

    def start(self):
        """
        Initializes tracking components using the provided screen dimensions.
        Prepares for calibration or gaze estimation.

        Raises:
            RuntimeError: If essential components fail to initialize or screen dimensions missing.
        """
        if self._is_running:
            logger.warning("API is already running. Call stop() first if you need to restart.")
            return

        logger.info("Starting GazeTrackerAPI resources...")
        # Ensure monitor dimensions are set (should be from __init__)
        if not self.monitor or 'width' not in self.monitor or 'height' not in self.monitor:
             # This should ideally not happen due to __init__ check, but good practice
             raise RuntimeError("Monitor dimensions not set during initialization.")
        logger.info(f"Using Screen Size: {self.monitor['width']}x{self.monitor['height']}")

        try:
            # 1. Initialize Gaze Tracking (MediaPipe)
            logger.info("Initializing GazeTracking (MediaPipe)...")
            self.gaze_tracking = GazeTracking()
            if self.gaze_tracking is None or self.gaze_tracking.face_mesh is None:
                 raise RuntimeError("Failed to initialize GazeTracking (MediaPipe).")

            # 2. Initialize Calibration and PointOfGaze objects
            #    These now rely on self.monitor being correctly set in __init__
            logger.info("Initializing GazeCalibration...")
            # Pass the self.monitor dict we stored
            self.gaze_calibration = GazeCalibration(self.gaze_tracking, self.monitor)
            if self.gaze_calibration is None:
                 raise RuntimeError("Failed to initialize GazeCalibration.")

            logger.info("Initializing PointOfGaze...")
             # Pass the self.monitor dict we stored
            self.pog = PointOfGaze(self.gaze_tracking, self.gaze_calibration, self.monitor, self.stabilize)
            if self.pog is None:
                 raise RuntimeError("Failed to initialize PointOfGaze.")

            # 3. Attempt to load calibration if a file path was provided
            if self.calibration_file_path and os.path.exists(self.calibration_file_path):
                # Pass self.gaze_calibration explicitly to load_calibration
                self.load_calibration(self.calibration_file_path)

            self._is_running = True
            logger.info("GazeTrackerAPI started successfully.")

        except Exception as e:
            logger.error(f"Error during API start: {e}")
            self.stop() # Attempt cleanup even if start failed partially
            raise # Re-raise the exception so the caller knows it failed


    def load_calibration(self, filepath):
        """
        Loads calibration parameters from a file.

        Args:
            filepath (str): The path to the calibration data file.

        Returns:
            bool: True if loading was successful, False otherwise.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            RuntimeError: If the API has not been started or GazeCalibration component is missing.
            ValueError: If the loaded data is invalid or missing required keys.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before loading calibration.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        if not self.gaze_calibration:
             # This check should be redundant if start() succeeded, but defensive coding helps.
             raise RuntimeError("GazeCalibration object not initialized. Cannot load data.")

        logger.info(f"Loading calibration data from {filepath}...")
        try:
            data = np.load(filepath)
            if 'poly_x' not in data or 'poly_y' not in data:
                raise ValueError("Calibration file is missing required keys ('poly_x', 'poly_y').")

            # --- Check Screen Size Compatibility ---
            if 'screen_width' in data and 'screen_height' in data:
                loaded_w, loaded_h = data['screen_width'], data['screen_height']
                current_w, current_h = self.monitor['width'], self.monitor['height']
                if int(loaded_w) != current_w or int(loaded_h) != current_h:
                    logger.warning(f"Loaded calibration screen size ({loaded_w}x{loaded_h}) "
                                   f"differs from current API screen size ({current_w}x{current_h}). "
                                   "Results may be inaccurate.")
            # --- End Screen Size Check ---

            # Assign loaded data to the GazeCalibration object
            self.gaze_calibration.poly_x = data['poly_x']
            self.gaze_calibration.poly_y = data['poly_y']
            self._is_calibrated = True
            self.calibration_file_path = filepath # Update path if loaded successfully
            logger.info("Calibration data loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration data from {filepath}: {e}")
            self._is_calibrated = False # Ensure state is correct on failure
            if not isinstance(e, (FileNotFoundError, ValueError)):
                 raise RuntimeError(f"Failed to load calibration data: {e}")
            else:
                 raise # Re-raise FileNotFoundError or ValueError


    def save_calibration(self, filepath):
        """
        Saves the current calibration parameters and screen dimensions to a file.

        Args:
            filepath (str): The path where the calibration data will be saved.

        Raises:
            RuntimeError: If the system is not calibrated yet or components missing.
            IOError: If saving the file fails.
        """
        if not self._is_calibrated:
            raise RuntimeError("System is not calibrated. Cannot save calibration data.")
        if not self.gaze_calibration or not hasattr(self.gaze_calibration, 'poly_x') or not hasattr(self.gaze_calibration, 'poly_y'):
             raise RuntimeError("Calibration data (polynomials) not found.")
        if not self.monitor:
             raise RuntimeError("Monitor dimensions not available for saving.")


        logger.info(f"Saving calibration data to {filepath}...")
        try:
            # Ensure the directory exists
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
            logger.info("API is not running.")
            return

        logger.info("Stopping GazeTrackerAPI resources...")

        # Potentially add cleanup for MediaPipe if needed (usually handled by Python's GC)
        if self.gaze_tracking and hasattr(self.gaze_tracking, 'face_mesh') and hasattr(self.gaze_tracking.face_mesh, 'close'):
             try:
                 self.gaze_tracking.face_mesh.close()
                 logger.info("MediaPipe FaceMesh resources released.")
             except Exception as e:
                 logger.warning(f"Exception while closing MediaPipe FaceMesh: {e}")

        # Reset internal state
        self.monitor = None
        self.gaze_tracking = None
        self.gaze_calibration = None
        self.pog = None
        # Keep calibration status, but mark as not running and not calibrating
        self._is_running = False
        self._is_calibrating = False # Ensure calibration stops if API stops
        logger.info("GazeTrackerAPI stopped.")

    def start_calibration(self, calibration_params=None):
        """
        Resets and starts the interactive calibration process, optionally applying
        custom parameters.

        Args:
            calibration_params (dict, optional): A dictionary to override default
                calibration parameters. Keys can include:
                'circle_radius', 'instruction_frames', 'fixation_frames',
                'calibration_frames', 'test_frames', 'run_test_stage' (bool).
                Defaults to None (use defaults).

        Returns:
            dict: The initial status dictionary from the calibration process.

        Raises:
            RuntimeError: If the API is not running or components are missing.
            TypeError: If calibration_params is not a dictionary when provided.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before calibration.")
        if not self.gaze_calibration or not self.pog:
             raise RuntimeError("Calibration components not initialized.")

        # --- Apply custom parameters before resetting ---
        if calibration_params is not None:
             if not isinstance(calibration_params, dict):
                  raise TypeError("calibration_params must be a dictionary or None.")
             try:
                  updated = self.gaze_calibration.set_parameters(calibration_params)
                  if updated:
                       logger.info(f"Applied custom calibration parameters: {calibration_params}")
                       logger.info(f"Current parameters: {self.gaze_calibration.get_current_parameters()}")
             except Exception as e:
                  # Catch potential errors during parameter setting
                  logger.error(f"Error applying calibration parameters: {e}")
                  # Should we proceed with defaults or raise? Let's proceed with potentially defaults/partially updated.
        # --- End parameter application ---

        # Reset calibration using the (potentially updated) parameters
        self.gaze_calibration.reset_calibration()

        self._is_calibrating = True
        self._is_calibrated = False # Start of calibration means not calibrated yet
        logger.info("Calibration process started/restarted.")

        # Manually construct the first status update based on current state
        initial_status = self.gaze_calibration._get_status_update(
             status='calibrating',
             phase='instruction', # Reset sets phase to instruction
             display_type='instruction_text',
             # Generate text based on current stage (which is 'ratios' after reset)
             text=f"Calibration: {self.gaze_calibration.current_stage.capitalize()} Stage\n\nPrepare to look at the dots.",
             phase_progress=0.0
        )
        return initial_status

    def process_calibration_step(self, frame):
        """
        Processes one frame/step of the calibration sequence.

        The caller should provide the current camera frame and use the returned
        dictionary to update their display (showing instructions, dots, etc.).

        Args:
            frame (numpy.ndarray): The current camera frame (BGR format).

        Returns:
            dict: A status dictionary describing the current calibration state
                  and what needs to be displayed. Contains keys like:
                  - 'status': ('calibrating', 'calculating', 'finished_all', 'error')
                  - 'stage', 'phase', 'progress'
                  - 'display_info': {'type', 'text', 'target_point', 'estimated_gaze'}
                  - 'final_errors': (Optional, included when status is 'finished_all')

        Raises:
            RuntimeError: If the API is not running or calibration was not started.
            ValueError: If the input frame is invalid.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() first.")
        if not self._is_calibrating:
            raise RuntimeError("Calibration not started. Call start_calibration() first.")
        if not self.gaze_calibration or not self.pog or not self.gaze_tracking:
             raise RuntimeError("Calibration components not initialized.")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame provided to process_calibration_step.")

        # 1. Refresh GazeTracking with the frame (updates landmarks, ratios)
        self.gaze_tracking.refresh(frame)

        # 2. Process the step using GazeCalibration
        #    It uses the updated info from gaze_tracking and the pog instance
        status_update = self.gaze_calibration.process_frame(self.pog)

        # 3. Update API state based on returned status
        returned_status = status_update.get('status')
        if returned_status == 'finished_all':
            logger.info("Calibration process finished successfully.")
            self._is_calibrated = True  # Calibration succeeded
            self._is_calibrating = False # Calibration process is no longer active
        elif returned_status == 'error':
             logger.error(f"Calibration process failed. Reason: {status_update.get('display_info', {}).get('text', 'Unknown error')}")
             self._is_calibrated = False # Calibration failed
             self._is_calibrating = False # Calibration process is no longer active
        elif self.gaze_calibration.is_finished():
             # Handle cases where GazeCalibration might finish but status isn't 'finished_all' (belt-and-suspenders)
             logger.warning("GazeCalibration is finished, but status was not 'finished_all'. Setting API state.")
             self._is_calibrated = (self.gaze_calibration.poly_x is not None) # Check if mapping was computed
             self._is_calibrating = False


        # 4. Return the status dictionary to the caller
        return status_update

    def cancel_calibration(self):
        """
        Stops the currently active calibration process.
        Calibration state (polynomials) will be invalid.
        """
        if not self._is_calibrating:
            logger.info("No active calibration process to cancel.")
            return

        logger.warning("Cancelling calibration process.")
        self._is_calibrating = False
        self._is_calibrated = False # Calibration was not completed successfully
        self.gaze_calibration.reset_calibration()
        logger.info("Calibration cancelled.")

    def get_gaze(self, frame):
        """
        Estimates the gaze position on the screen for a given camera frame.

        Args:
            frame (numpy.ndarray): The current camera frame (BGR format typically,
                                   but GazeTracking handles conversion if needed).

        Returns:
            tuple[int | None, int | None]: The estimated (x, y) screen coordinates
                                           of the gaze, or (None, None) if not calibrated
                                           or gaze cannot be determined for this frame.

        Raises:
            RuntimeError: If the API has not been started using start().
            ValueError: If the input frame is invalid.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before getting gaze.")
        if not self._is_calibrated:
            # No longer logging a warning every frame, return None quietly
            # logger.warning("Calibration not performed or loaded. Cannot estimate gaze.")
            return (None, None)
        if not self.gaze_tracking or not self.pog:
             logger.warning("Tracking/POG components not initialized. Cannot get gaze.")
             return (None, None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame provided to get_gaze.")

        # 1. Refresh GazeTracking with the new frame
        #    This updates landmarks, eye objects, pupil info, etc.
        self.gaze_tracking.refresh(frame)

        # 2. Calculate the point of gaze using the PointOfGaze object
        #    This uses the latest tracking info and the loaded/calculated calibration
        estimated_gaze = self.pog.point_of_gaze() # Returns (x, y) or (None, None)

        # 3. Return the result
        return estimated_gaze
    
    def get_landmarks(self, frame):
        """
        Processes the frame and returns the detected MediaPipe face landmarks.

        Args:
            frame (numpy.ndarray): The current camera frame (BGR format typically,
                                   but GazeTracking handles conversion if needed).

        Returns:
            list | None: A list of MediaPipe NormalizedLandmark objects for the
                         detected face, or None if no face is detected or the
                         API is not running. The landmarks are normalized coordinates
                         (x, y, z) within the frame dimensions.

        Raises:
            RuntimeError: If the API has not been started using start().
            ValueError: If the input frame is invalid.
        """
        if not self._is_running:
            raise RuntimeError("API must be started with start() before getting landmarks.")
        if not self.gaze_tracking:
             logger.warning("GazeTracking not initialized. Cannot get landmarks.")
             return None
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame provided to get_landmarks.")

        # Process the frame using the existing refresh method in GazeTracking
        self.gaze_tracking.refresh(frame)

        # Access the stored landmarks from GazeTracking
        # GazeTracking._analyze_frame sets self.landmarks if a face is found
        if hasattr(self.gaze_tracking, 'landmarks') and self.gaze_tracking.landmarks:
            # Return the raw list of NormalizedLandmark objects
            return self.gaze_tracking.landmarks
        else:
            # No face/landmarks detected in this frame
            return None
        
    def get_calibration_test_data(self):
        """
        Retrieves the detailed gaze and landmark data collected during the last
        completed test stage of calibration.

        Returns:
            tuple[list, list] | tuple[None, None]:
                A tuple containing (gaze_data, landmark_data), where:
                - gaze_data: List of (frame_idx, point_idx, target_x, target_y, est_x, est_y)
                - landmark_data: List of (frame_idx, point_idx, landmarks_list)
                                (landmarks_list is list of [x,y,z])
                Returns (None, None) if calibration hasn't been completed or
                GazeCalibration object doesn't exist.
        """
        if not self._is_calibrated:
            logger.warning("Calibration not completed. No test data available.")
            return None, None
        if not self.gaze_calibration:
            logger.error("GazeCalibration object not available.")
            return None, None

        gaze_data = self.gaze_calibration.get_test_stage_gaze_data()
        landmark_data = self.gaze_calibration.get_test_stage_landmark_data()
        return gaze_data, landmark_data

    def save_calibration_test_data(self, gaze_filepath="calibration_gaze_data.csv", landmarks_filepath="calibration_landmark_data.csv"):
        """
        Saves the detailed gaze and landmark data collected during the last
        completed test stage of calibration (if it was run) to CSV files.

        Args:
            gaze_filepath (str): Path to save the gaze data CSV file.
            landmarks_filepath (str): Path to save the landmark data CSV file.
                                      Note: Saving landmarks can create large files.

        Returns:
            bool: True if both files were saved successfully (or if data was empty),
                  False otherwise.

        Raises:
            RuntimeError: If calibration was not completed or components missing.
        """

        gaze_data, landmark_data = self.get_calibration_test_data()

        if gaze_data is None and landmark_data is None:
            raise RuntimeError("Cannot save test data: Calibration not completed or components missing.")

        if not gaze_data and not landmark_data:
             logger.info("No test stage data found (likely skipped). Nothing to save.")
             return True # Return True as there's nothing to do/fail

        success_gaze = True
        success_landmarks = True

        # --- Save Gaze Data ---
        if gaze_data:
            try:
                logger.info(f"Saving gaze test data ({len(gaze_data)} samples) to {gaze_filepath}...")
                df_gaze = pd.DataFrame(gaze_data, columns=['frame_idx', 'point_idx', 'target_x', 'target_y', 'gaze_x', 'gaze_y'])
                os.makedirs(os.path.dirname(gaze_filepath) or '.', exist_ok=True)
                df_gaze.to_csv(gaze_filepath, index=False)
                logger.info("Gaze test data saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save gaze test data to {gaze_filepath}: {e}")
                success_gaze = False
        else:
             # This case is now handled by the initial check
             # logger.info("No gaze test data collected to save.")
             pass

        # --- Save Landmark Data --- (only if list is not empty)
        if landmark_data:
            try:
                 logger.info(f"Saving landmark test data ({len(landmark_data)} samples) to {landmarks_filepath}...")
                 landmark_rows = []
                 for frame_idx, point_idx, landmarks_list in landmark_data:
                     for lm_idx, lm_coords in enumerate(landmarks_list):
                          landmark_rows.append([frame_idx, point_idx, lm_idx, lm_coords[0], lm_coords[1], lm_coords[2]])

                 if landmark_rows:
                      df_landmarks = pd.DataFrame(landmark_rows, columns=['frame_idx', 'point_idx', 'landmark_id', 'x', 'y', 'z'])
                      os.makedirs(os.path.dirname(landmarks_filepath) or '.', exist_ok=True)
                      df_landmarks.to_csv(landmarks_filepath, index=False)
                      logger.info("Landmark test data saved successfully.")
                 else:
                      logger.info("Landmark data list was empty after processing.")
            except Exception as e:
                logger.error(f"Failed to save landmark test data to {landmarks_filepath}: {e}")
                success_landmarks = False
        else:
             # This case is now handled by the initial check
             # logger.info("No landmark test data collected to save.")
             pass

        return success_gaze and success_landmarks
    
    def save_calibration(self, filepath):
        """
        Saves the current calibration parameters to a file.

        Args:
            filepath (str): The path where the calibration data will be saved.

        Raises:
            RuntimeError: If the system is not calibrated yet.
            IOError: If saving the file fails.
        """
        if not self._is_calibrated:
            raise RuntimeError("System is not calibrated. Cannot save calibration data.")
        if not self.gaze_calibration or not hasattr(self.gaze_calibration, 'poly_x') or not hasattr(self.gaze_calibration, 'poly_y'):
             raise RuntimeError("Calibration data (polynomials) not found.")

        logger.info(f"Saving calibration data to {filepath}...")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Save the polynomial coefficients
            np.savez(
                filepath,
                poly_x=self.gaze_calibration.poly_x,
                poly_y=self.gaze_calibration.poly_y
                )
            logger.info("Calibration data saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save calibration data to {filepath}: {e}")
            raise IOError(f"Failed to save calibration data: {e}")


    def load_calibration(self, filepath):
        """
        Loads calibration parameters from a file.

        Args:
            filepath (str): The path to the calibration data file.

        Returns:
            bool: True if loading was successful, False otherwise.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            RuntimeError: If the API has not been started or components are missing.
            ValueError: If the loaded data is invalid or missing required keys.
        """
        if not self._is_running:
            # Allow loading before start() if we adjust initialization, but safer to require start() first
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

            # Assign loaded data to the GazeCalibration object
            self.gaze_calibration.poly_x = data['poly_x']
            self.gaze_calibration.poly_y = data['poly_y']
            self._is_calibrated = True
            self.calibration_file_path = filepath # Update path if loaded successfully
            logger.info("Calibration data loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration data from {filepath}: {e}")
            self._is_calibrated = False # Ensure state is correct on failure
            if not isinstance(e, (FileNotFoundError, ValueError)):
                 raise RuntimeError(f"Failed to load calibration data: {e}")
            else:
                 raise # Re-raise FileNotFoundError or ValueError

    @property
    def is_calibrated(self):
        """
        Checks if the system has been successfully calibrated or had calibration data loaded.

        Returns:
            bool: True if calibrated, False otherwise.
        """
        return self._is_calibrated

    @property
    def is_running(self):
        """
        Checks if the API resources (camera, trackers) have been started.

        Returns:
            bool: True if running, False otherwise.
        """
        return self._is_running
    
    @property
    def is_calibrating(self):
        """Checks if the calibration process is currently active."""
        return self._is_calibrating

    # Context Manager Protocol (Optional but nice for resource management)
    def __enter__(self):
        """Enables use with 'with' statement."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures stop() is called when exiting 'with' block."""
        self.stop()
        # Return False to propagate exceptions, True to suppress them (usually False is better)
        return False
    