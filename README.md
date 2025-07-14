# Gaze Tracking API

## Overview

This project provides a Python API (`GazeTrackerAPI`) for webcam-based gaze tracking and calibration. It uses MediaPipe for facial landmark detection and calculates a mapping between eye gaze direction and screen coordinates.

The primary goal is to enable the integration of gaze tracking into larger applications, such as scientific experiments investigating attention bias by presenting stimuli and recording corresponding gaze projections.

Facilitates saving of:
    *   Calibration model parameters (`.npz`).
    *   Detailed gaze and landmark data from the calibration test stage (`.csv`).
    *   Gaze and landmark data during custom evaluation tasks (`.csv`).

Designed to work with frames provided by the caller (e.g., from a live webcam or a pre-recorded video file).

**⚠️ Disclaimer: Calibration Save/Load**
*The functionality for saving (`save_calibration`) and loading (`load_calibration`) calibration settings (`.npz` file) is just an experiment from me and is not reliable. Please perform a fresh calibration for each session.*

## Setup (macOS + brew + conda)

1.  Create and activate conda environment:
    ```bash
    conda create -n gaze-env python=3.12 anaconda
    conda activate gaze-env
    ```
2.  Install PortAudio (dependency for PyAudio, which might be in requirements):
    ```bash
    brew install portaudio
    ```
3.  Install Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Core API: `GazeTrackerAPI`

The main interface is the `GazeTrackerAPI` class located in `src/api.py`. It encapsulates the necessary components for tracking, calibration, and point-of-gaze calculation.

### `GazeTrackerAPI` methods

*   **`__init__(self, screen_width, screen_height, stabilize_gaze=True, calibration_file=None)`**:
    *   Initializes the API configuration. Requires screen dimensions.
    *   `stabilize_gaze`: Enables/disables temporal smoothing of gaze estimates.
    *   `calibration_file`: Optional path to an `.npz` file to load pre-saved calibration.
*   **`start(self)`**: Initializes internal components (MediaPipe, etc.) and loads calibration if `calibration_file` was provided and exists. Must be called before calibration or gaze estimation. Automatically called when using `with` statement.
*   **`stop(self)`**: Releases resources. Automatically called when exiting `with` statement.
*   **`start_calibration(self, calibration_params=None)`**:
    *   Resets and begins the interactive calibration process.
    *   `calibration_params`: Optional dictionary to override default timings, dot radius, and whether to run the test stage (see Configuration section below).
    *   Returns the initial status dictionary for UI display.
*   **`process_calibration_step(self, frame)`**:
    *   Processes a single frame during active calibration.
    *   Requires the current camera/video `frame` (NumPy array) used for gaze tracking.
    *   Returns a status dictionary detailing the current stage, phase, progress, and necessary UI display information (`display_info`).
*   **`cancel_calibration(self)`**: Stops the active calibration process. Calibration will be incomplete.
*   **`get_gaze(self, frame)`**:
    *   Estimates gaze coordinates for the given `frame`.
    *   Requires prior successful calibration.
    *   Processes the frame using MediaPipe (updates internal landmarks).
    *   Returns `(x, y)` tuple or `(None, None)`.
*   **`get_landmarks(self, frame)`**:
    *   Processes the `frame` using MediaPipe.
    *   Returns the list of raw MediaPipe `NormalizedLandmark` objects for the detected face, or `None`.
*   **`save_calibration(self, filepath)`**: Saves the computed calibration polynomials and screen dimensions to an `.npz` file. Requires calibration to be complete.
*   **`load_calibration(self, filepath)`**: Loads calibration polynomials from an `.npz` file. Requires `start()` to have been called.
*   **`get_calibration_test_data(self)`**: Retrieves detailed gaze and landmark data collected during the calibration's test stage (if run). Returns `(gaze_data_list, landmark_data_list)`.
*   **`save_calibration_test_data(self, gaze_filepath="...", landmarks_filepath="...")`**: Saves the calibration test stage data to specified CSV files.
*   **`is_calibrated` (Property)**: Returns `True` if calibration is complete/loaded, `False` otherwise.
*   **`is_calibrating` (Property)**: Returns `True` if `start_calibration` has been called and the process is not yet finished or cancelled.
*   **`is_running` (Property)**: Returns `True` if `start()` has been called successfully and `stop()` hasn't been called.

## Configuration

### API Initialization (`__init__`)

*   `screen_width`, `screen_height` (int, required): Pixel dimensions of the display screen.
*   `stabilize_gaze` (bool, optional, default=`True`): Apply smoothing to gaze output.
*   `calibration_file` (str, optional, default=`None`): Path to `.npz` file for loading calibration.

### Calibration Parameters (`start_calibration`)

Passed as a dictionary to the `calibration_params` argument:

*   `circle_radius` (int, optional): Radius used for generating calibration target points. Default: `20`.
*   `instruction_frames` (int, optional): Duration (in frames) to display stage introduction text. Default: `40`.
*   `fixation_frames` (int, optional): Duration to display a point before data collection starts. Default: `20`.
*   `calibration_frames` (int, optional): Duration for collecting data per point (ratios stage). Default: `40`.
*   `test_frames` (int, optional): Duration for collecting data per point (test stage). Default: `40`.
*   `run_test_stage` (bool, optional): Whether to run the test/validation stage after core calibration. Default: `True`.

## Data Output

*   **Calibration Settings (`.npz`)**: Saved via `save_calibration`. Contains:
    *   `poly_x`, `poly_y`: NumPy arrays holding the polynomial coefficients.
    *   `screen_width`, `screen_height`: Dimensions used for this calibration.
*   **Calibration Test Gaze Data (`.csv`)**: Saved via `save_calibration_test_data`. Columns:
    *   `frame_idx`: Frame index within the test stage.
    *   `point_idx`: Index of the calibration point being tested.
    *   `target_x`, `target_y`: Coordinates of the target point.
    *   `gaze_x`, `gaze_y`: Estimated gaze coordinates for that frame.
*   **Calibration Test Landmark Data (`.csv`)**: Saved via `save_calibration_test_data`. Columns:
    *   `frame_idx`, `point_idx`: See above.
    *   `landmark_id`: Index of the landmark (0-477 for MediaPipe Face Mesh with Iris).
    *   `x`, `y`, `z`: Normalized coordinates of the landmark for that frame.
*   **Evaluation Gaze/Landmark Data (`.csv`)**: Saved via helper functions in the example script (`save_evaluation_data`). Typically contains:
    *   `stimulus_frame`: Frame number of the displayed stimulus video.
    *   `gaze_x`, `gaze_y`: Estimated gaze coordinates (can be NaN if not detected).
    *   *(Evaluation Landmarks CSV)*: `stimulus_frame`, `landmark_id`, `x`, `y`, `z`.

## Running the Example

The `examples/api_test.py` script provides a comprehensive demonstration:

1.  **Configure:** Edit the flags and paths in the `### --- Configuration --- ###` section of `api_test.py` (e.g., `USE_WEBCAM_FOR_GAZE`, `RUN_ATTENTION_EVALUATION`, video paths).
2.  **Run:** Execute the script from the project root directory (`airest_cv`):
    ```bash
    python examples/api_test.py
    ```
3.  **Interact:** Follow the on-screen calibration instructions (focus on dots). Press `ESC` to cancel calibration or exit stages.
4.  **Check Results:** Output files (`.npz`, `.csv`) will be saved in the directory specified by `RESULTS_DIR_NAME` (default: `api_test_results_final`).

# Basic Usage Workflow (Extended version is provided in `examples/api_test.py`!)

```python
import cv2
from src.api import GazeTrackerAPI
# Assume screen_width, screen_height are known
# Assume get_frame() reads a frame from your video source (webcam/file)

SCREEN_W = 1920
SCREEN_H = 1080
CALIB_FILE = "calibration_results/calibration_settings.npz" # Optional

api = None # Define outside try for finally clause
try:
    # Use 'with' for automatic resource management (start/stop)
    with GazeTrackerAPI(screen_width=SCREEN_W, screen_height=SCREEN_H,
                        # calibration_file=CALIB_FILE # Uncomment to load
                       ) as api:

        print(f"API Running: {api.is_running}")

        # --- Calibration (if not loaded) ---
        if not api.is_calibrated:
            print("Starting calibration...")
            # Customize parameters if needed
            calib_params = {'run_test_stage': True, 'instruction_frames': 50}
            status = api.start_calibration(calibration_params=calib_params)

            while api.is_calibrating:
                frame = get_frame() # Get frame from your source
                if frame is None: break
                # frame = cv2.flip(frame, 1) # Optional flip

                status = api.process_calibration_step(frame)

                # --- Caller responsibilty: Display calibration UI ---
                # Based on status['display_info']: type, text, target_point, etc.
                # e.g., draw_calibration_display(SCREEN_W, SCREEN_H, status['display_info'])
                # Handle cv2.waitKey() & ESC to call api.cancel_calibration()
                # --- End display logic ---

                if status['status'] == 'finished_all' or status['status'] == 'error':
                    break # Exit calibration loop

            if api.is_calibrated:
                print("Calibration successful!")
                api.save_calibration(CALIB_FILE) # Save successful calibration
                # Optionally save test data if run_test_stage was True
                if calib_params.get('run_test_stage', True):
                     api.save_calibration_test_data(
                         gaze_filepath="calibration_results/calib_gaze.csv",
                         landmarks_filepath="calibration_results/calib_landmarks.csv"
                     )
            else:
                print("Calibration failed or cancelled.")
                exit() # Or handle error appropriately

        # --- Gaze Estimation ---
        print("Starting gaze estimation...")
        while True:
            frame = get_frame() # Get frame from your source
            if frame is None: break
            # frame = cv2.flip(frame, 1) # Optional flip

            gaze_xy = api.get_gaze(frame)
            landmarks = api.get_landmarks(frame) # Also processes the frame internally

            if gaze_xy and gaze_xy[0] is not None:
                print(f"Gaze: ({gaze_xy[0]}, {gaze_xy[1]})")
                # --- Caller responsibility: Use gaze_xy ---
                # e.g., draw gaze point on display_frame
            else:
                print("Gaze: N/A")

            # Handle cv2.waitKey() & ESC exit

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
finally:
    # Cleanup is handled by 'with' statement (calls api.stop())
    print("API stopped.")

```

## Web Demo

The repository includes a simple FastAPI application that demonstrates a basic
attention-bias test. To run it locally:

```bash
python webapp.py
```

The application selects an available port automatically and binds to all
network interfaces. Check the console output for the URL and open it in your
browser to start the calibration and slideshow process.
