### --- Imports and Setup --- ###

import cv2
import numpy as np
import os
import sys
import time
import traceback
import pandas as pd

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
project_root = parent_dir
sys.path.append(parent_dir)

# --- API and Utilities ---
try:
    from src.api import GazeTrackerAPI
    from src.utils.utils import get_screen_size
except ImportError as e:
    print(f"ERROR: Failed to import required modules. Ensure src is in PYTHONPATH. Details: {e}")
    sys.exit(1)

# examples/api_calibration_test.py (Configuration Section)

### --- Configuration --- ###

# --- Input Sources ---
USE_WEBCAM_FOR_GAZE: bool = False # True for live webcam, False for video file
WEBCAM_ID: int = 0
# Path relative to project root
GAZE_INPUT_VIDEO_PATH: str = os.path.join(project_root, "test_videos", "test_webcam.mp4") # Corrected Path

# Source for Stimulus Presentation (Displayed Fullscreen during Evaluation)
STIMULUS_VIDEO_PATH: str = os.path.join(project_root, "eval_videos", "Video_test_3_20s.mp4") # Corrected Path

# --- Control Flags ---
LOAD_CALIBRATION: bool = False
RUN_CALIBRATION_TEST_STAGE: bool = True
RUN_ATTENTION_EVALUATION: bool = False
RUN_SIMPLE_GAZE_TEST: bool = True

# --- API Settings ---
STABILIZE_GAZE: bool = True

# --- Optional: Custom Calibration Parameters ---
CUSTOM_CALIBRATION_PARAMS: dict = {
    'instruction_frames': 50, 'fixation_frames': 15, 'calibration_frames': 30,
    'test_frames': 30, 'circle_radius': 25, 'run_test_stage': RUN_CALIBRATION_TEST_STAGE
}
USE_CUSTOM_PARAMS: bool = False

# --- Output Files ---
RESULTS_DIR_NAME: str = "api_test_results" # Updated directory name
RESULTS_DIR: str = os.path.join(project_root, RESULTS_DIR_NAME)
CALIBRATION_NPZ_FILENAME: str = os.path.join(RESULTS_DIR, "calibration_settings.npz")
CALIB_GAZE_CSV_FILENAME: str = os.path.join(RESULTS_DIR, "calibration_test_gaze_data.csv")
CALIB_LANDMARKS_CSV_FILENAME: str = os.path.join(RESULTS_DIR, "calibration_test_landmarks.csv")
EVALUATION_GAZE_CSV_FILENAME: str = os.path.join(RESULTS_DIR, "evaluation_gaze_data.csv")
EVALUATION_LANDMARKS_CSV_FILENAME: str = os.path.join(RESULTS_DIR, "evaluation_landmarks.csv") # New output file for eval landmarks


### --- Helper Functions --- ###

def setup_environment():
    """Gets screen size, creates output directory. Returns screen dimensions."""
    print("--- Setting up environment ---")
    try: os.makedirs(RESULTS_DIR, exist_ok=True); print(f"Results directory ensured: {RESULTS_DIR}")
    except OSError as e: print(f"ERROR: Could not create results dir '{RESULTS_DIR}': {e}"); return None
    screen_size = get_screen_size()
    if not screen_size: print("ERROR: Could not get screen size."); return None
    print(f"Screen Size Detected: {screen_size['width']}x{screen_size['height']}")
    return screen_size['width'], screen_size['height']

def initialize_video_source(source_description, is_file):
    """Opens a video source, returns cap object and properties."""
    print(f"--- Initializing video source: {source_description} ---")
    source_input = source_description
    if is_file:
        abs_path = os.path.abspath(source_description)
        print(f"    Attempting file path: {abs_path}")
        if not os.path.exists(abs_path): print(f"ERROR: Video file not found at: {abs_path}"); return None, 0, 0, 0
        source_input = abs_path
    else: # It's a webcam ID
         print(f"    Attempting webcam ID: {source_description}")
         source_input = int(source_description) # Ensure it's an int

    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened(): print(f"ERROR: Failed to open video source: {source_description}"); return None, 0, 0, 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"    Source opened: {width}x{height}" + (f", FPS: {fps:.2f}" if fps > 0 else ""))
    return cap, width, height, fps

def rewind_video_capture(cap, source_desc="Video source"):
    """Rewinds a video capture object if it's valid."""
    if cap and cap.isOpened():
        print(f"Rewinding {source_desc}...")
        success = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not success: print(f"WARN: Failed to rewind {source_desc}.")
        return success
    return False

def draw_calibration_display(width, height, display_info):
    """Creates the calibration UI frame (dots/text)."""
    # (Implementation unchanged)
    display_frame = np.zeros((height, width, 3), dtype=np.uint8)
    info_type = display_info.get('type')
    target_point = display_info.get('target_point')
    estimated_gaze = display_info.get('estimated_gaze')
    text = display_info.get('text')
    text_color, dot_color, gaze_dot_color = (255, 255, 255), (0, 0, 255), (200, 200, 200)
    dot_radius, gaze_dot_radius = DEFAULT_PARAMS["circle_radius"], 10
    if info_type in ('instruction_text', 'message') and text:
        y0, dy = 150, 50
        for i, line in enumerate(text.split('\n')):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.putText(display_frame, line, ((width - tw) // 2, y0 + i * (dy + th)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    elif info_type in ('fixation_dot', 'test_dot'):
        if target_point: cv2.circle(display_frame, tuple(target_point), dot_radius, dot_color, -1)
        if info_type == 'test_dot' and estimated_gaze:
            cv2.circle(display_frame, tuple(estimated_gaze), gaze_dot_radius, gaze_dot_color, -1)
    return display_frame

def draw_gaze_overlay(frame, gaze_history, screen_w, screen_h, cam_w, cam_h):
    """Draws gaze history on a frame (typically the gaze input frame)."""
    # (Implementation unchanged)
    overlay = frame.copy()
    history_len = len(gaze_history)
    text_color_ok, text_color_na = (0, 255, 0), (0, 0, 255)
    gaze_dot_color, gaze_border_color = (0, 0, 255), (255, 255, 255)
    if not gaze_history:
        cv2.putText(overlay, "Gaze: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color_na, 2)
        return overlay
    for i, pt in enumerate(gaze_history):
        hist_x = int((pt[0] / screen_w) * cam_w); hist_y = int((pt[1] / screen_h) * cam_h)
        alpha = (i + 1) / history_len
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))
        cv2.circle(overlay, (hist_x, hist_y), 5, color, -1)
    curr_gaze_x, curr_gaze_y = gaze_history[-1]
    draw_x = int((curr_gaze_x / screen_w) * cam_w); draw_y = int((curr_gaze_y / screen_h) * cam_h)
    cv2.circle(overlay, (draw_x, draw_y), 8, gaze_dot_color, -1)
    cv2.circle(overlay, (draw_x, draw_y), 10, gaze_border_color, 1)
    cv2.putText(overlay, f"Gaze: ({curr_gaze_x}, {curr_gaze_y})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color_ok, 2)
    return overlay

def save_calibration_results(api):
    """Saves calibration settings and test data (if run)."""
    print("--- Saving calibration results ---")
    save_ok = True
    try:
        api.save_calibration(CALIBRATION_NPZ_FILENAME)
        print(f"* Calibration settings saved: {CALIBRATION_NPZ_FILENAME}")
    except Exception as e: print(f"ERROR saving calibration settings: {e}"); save_ok = False

    if RUN_CALIBRATION_TEST_STAGE:
        try:
             print("Attempting to save calibration test stage data...")
             save_success_test = api.save_calibration_test_data(
                  gaze_filepath=CALIB_GAZE_CSV_FILENAME, landmarks_filepath=CALIB_LANDMARKS_CSV_FILENAME)
             if save_success_test:
                  gaze_data, landmark_data = api.get_calibration_test_data()
                  if gaze_data: print(f"* Calib test gaze saved: {CALIB_GAZE_CSV_FILENAME}")
                  if landmark_data: print(f"* Calib test landmarks saved: {CALIB_LANDMARKS_CSV_FILENAME}")
                  if not gaze_data and not landmark_data: print("* Calib test stage ran, but no data collected/saved.")
             else: print("WARN: Failed saving calib test data file(s)."); save_ok = False
        except Exception as e: print(f"ERROR saving calib test data: {e}"); traceback.print_exc(); save_ok = False
    else: print("Skipped saving calibration test stage data.")
    return save_ok

def save_evaluation_data(gaze_data, landmark_data, gaze_filepath, landmark_filepath):
    """Saves the attention evaluation gaze and landmark data to CSV files."""
    gaze_ok, landmark_ok = True, True # Assume success unless proven otherwise

    # --- Save Gaze Data ---
    if not gaze_data:
        print("No evaluation gaze data collected to save.")
    else:
        print(f"--- Saving evaluation gaze results ({len(gaze_data)} samples) ---")
        try:
            df_gaze = pd.DataFrame(gaze_data, columns=['stimulus_frame', 'gaze_x', 'gaze_y'])
            os.makedirs(os.path.dirname(gaze_filepath) or '.', exist_ok=True)
            df_gaze.to_csv(gaze_filepath, index=False)
            print(f"* Evaluation gaze data saved: {gaze_filepath}")
        except Exception as e:
            print(f"ERROR saving evaluation gaze data to {gaze_filepath}: {e}")
            traceback.print_exc(); gaze_ok = False

    # --- Save Landmark Data ---
    if not landmark_data:
        print("No evaluation landmark data collected to save.")
    else:
        print(f"--- Saving evaluation landmark results ({len(landmark_data)} samples) ---")
        try:
            landmark_rows = []
            for frame_idx, landmarks_list in landmark_data: # Get frame index from tuple
                 # Assuming landmarks_list is already [ [x,y,z], [x,y,z], ... ]
                 for lm_idx, lm_coords in enumerate(landmarks_list):
                      landmark_rows.append([frame_idx, lm_idx, lm_coords[0], lm_coords[1], lm_coords[2]])

            if landmark_rows:
                 df_landmarks = pd.DataFrame(landmark_rows, columns=['stimulus_frame', 'landmark_id', 'x', 'y', 'z'])
                 os.makedirs(os.path.dirname(landmark_filepath) or '.', exist_ok=True)
                 df_landmarks.to_csv(landmark_filepath, index=False)
                 print(f"* Evaluation landmark data saved: {landmark_filepath}")
            else: print("Evaluation landmark data list was empty after processing.") # Should not happen if landmark_data is not empty
        except Exception as e:
            print(f"ERROR saving evaluation landmark data to {landmark_filepath}: {e}")
            traceback.print_exc(); landmark_ok = False

    return gaze_ok and landmark_ok


### --- Core Application Functions --- ###

def run_calibration_session(api, gaze_input_cap, screen_w, screen_h):
    """Runs the interactive calibration process using the gaze input source."""
    print("\n### --- Starting Calibration Session --- ###")
    params_to_use = CUSTOM_CALIBRATION_PARAMS if USE_CUSTOM_PARAMS else {}
    params_to_use['run_test_stage'] = RUN_CALIBRATION_TEST_STAGE
    print(f"Using parameters: {params_to_use}")

    calibration_status = api.start_calibration(calibration_params=params_to_use)
    if not calibration_status: print("ERROR: Failed to start API calibration."); return False

    window_name = "Calibration Display (Press ESC to Cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)

    while api.is_calibrating:
        ret, gaze_frame = gaze_input_cap.read() # Read frame FOR GAZE INPUT
        if not ret or gaze_frame is None:
            gaze_is_file = not USE_WEBCAM_FOR_GAZE
            if gaze_is_file: api.cancel_calibration(); print("ERROR: Gaze input video ended during calibration.")
            else: print("WARN: Cannot read frame from gaze input source."); time.sleep(0.1)
            break # Exit loop if input fails

        gaze_frame = cv2.flip(gaze_frame, 1)
        calibration_status = api.process_calibration_step(gaze_frame) # Pass GAZE frame to API
        display_frame = draw_calibration_display(screen_w, screen_h, calibration_status['display_info'])
        cv2.imshow(window_name, display_frame) # Show calibration UI

        key = cv2.waitKey(1) & 0xFF
        if key == 27: api.cancel_calibration(); print("ESC pressed, cancelling calibration."); break

    cv2.destroyWindow(window_name); cv2.waitKey(500)

    calibration_successful = api.is_calibrated
    if calibration_successful:
        final_text = calibration_status.get('display_info', {}).get('text', 'Calibration Complete')
        print(f"--- {final_text} ---")
        save_calibration_results(api)
        print("### --- Calibration Session End (Success) --- ###")
    else:
        print("--- Calibration Failed or Cancelled ---")
        print("### --- Calibration Session End (Failure/Cancelled) --- ###")
    return calibration_successful

def run_attention_evaluation(api, stimulus_cap, gaze_input_cap, screen_w, screen_h):
    """Plays stimulus video fullscreen, records gaze AND landmarks from gaze_input source."""
    print("\n### --- Starting Attention Evaluation Session --- ###")
    eval_gaze_results = []
    eval_landmark_results = [] # New list for landmarks
    stimulus_fps = stimulus_cap.get(cv2.CAP_PROP_FPS)
    stimulus_frame_count = int(stimulus_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Stimulus video: {stimulus_fps:.2f} FPS, {stimulus_frame_count} frames (approx)")

    # --- Fix Fullscreen ---
    window_name = "Attention Evaluation Task (Press ESC to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)

    stimulus_frame_num = 0
    while True:
        # Read Stimulus Video Frame (for display)
        ret_stim, frame_stim = stimulus_cap.read()
        if not ret_stim or frame_stim is None: print("End of stimulus video."); break
        stimulus_frame_num = int(stimulus_cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Read Corresponding Gaze Input Frame (for API)
        ret_gaze, frame_gaze = gaze_input_cap.read()
        gaze_coords = (None, None)
        landmarks = None # Variable to hold landmarks
        if not ret_gaze or frame_gaze is None:
            gaze_is_file = not USE_WEBCAM_FOR_GAZE
            if gaze_is_file: print("ERROR: Gaze input video ended during evaluation."); break
            else: print("WARN: Cannot read gaze input. Recording NaN."); gaze_coords = (None, None)
        else:
            frame_gaze = cv2.flip(frame_gaze, 1)
            # --- Get Gaze AND Landmarks ---
            gaze_coords = api.get_gaze(frame_gaze)
            landmarks_mp = api.get_landmarks(frame_gaze) # Use the existing API method
            if landmarks_mp:
                 # Convert to savable format immediately
                 landmarks = [[lm.x, lm.y, lm.z] for lm in landmarks_mp]

        # --- Record Data ---
        gaze_x = gaze_coords[0] if gaze_coords and gaze_coords[0] is not None else np.nan
        gaze_y = gaze_coords[1] if gaze_coords and gaze_coords[1] is not None else np.nan
        eval_gaze_results.append((stimulus_frame_num, gaze_x, gaze_y))
        if landmarks: # Only append if landmarks were found
            eval_landmark_results.append((stimulus_frame_num, landmarks))

        # Display Stimulus Frame
        cv2.imshow(window_name, frame_stim)

        # Handle Input and Delay
        delay = max(1, int(1000 / stimulus_fps)) if stimulus_fps > 0 else 1
        key = cv2.waitKey(delay) & 0xFF
        if key == 27: print("ESC pressed, stopping evaluation."); break

    cv2.destroyWindow(window_name); cv2.waitKey(1)
    # Save both gaze and landmark data
    save_evaluation_data(eval_gaze_results, eval_landmark_results,
                         EVALUATION_GAZE_CSV_FILENAME, EVALUATION_LANDMARKS_CSV_FILENAME)
    print(f"Processed {stimulus_frame_num} stimulus frames during evaluation.")
    print("### --- Attention Evaluation Session End --- ###")

def run_simple_gaze_test_session(api, gaze_input_cap, screen_w, screen_h, cam_w, cam_h, fps):
    """Runs gaze estimation, overlays results on the gaze input video feed."""
    print("\n### --- Starting Simple Gaze Test Session --- ###")
    window_name = "Simple Gaze Test (Overlay on Input - Press ESC to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(window_name, cam_w, cam_h)

    gaze_history = []; history_len = 15; frame_count = 0
    gaze_is_file = not USE_WEBCAM_FOR_GAZE

    while True:
        ret, frame = gaze_input_cap.read() # Read from GAZE source
        if not ret or frame is None:
            if gaze_is_file: print("End of gaze input video."); break
            else: print("WARN: Cannot read gaze input frame."); time.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        gaze_coords = api.get_gaze(frame) # Get gaze from GAZE source frame

        # Update history and draw overlay on GAZE source frame
        if gaze_coords and gaze_coords[0] is not None: gaze_history.append(gaze_coords)
        else: gaze_history.clear()
        if len(gaze_history) > history_len: gaze_history.pop(0)
        display_frame = draw_gaze_overlay(frame, gaze_history, screen_w, screen_h, cam_w, cam_h)

        cv2.imshow(window_name, display_frame) # Display the GAZE source frame with overlay

        delay = max(1, int(1000 / fps)) if fps > 0 and gaze_is_file else 1
        key = cv2.waitKey(delay) & 0xFF
        if key == 27: print("ESC pressed, stopping simple gaze test."); break
        frame_count += 1

    cv2.destroyWindow(window_name); cv2.waitKey(1)
    print(f"Processed {frame_count} frames during simple test.")
    print("### --- Simple Gaze Test Session End --- ###")


### --- Main Script --- ###

if __name__ == "__main__":
    print("\n======= Starting Gaze Tracker API Test Script =======")

    # --- Step 1: Setup Environment ---
    screen_dims = setup_environment()
    if screen_dims is None: sys.exit(1)
    SCREEN_WIDTH, SCREEN_HEIGHT = screen_dims

    # --- Step 2: Initialize Video Sources ---
    # Source for Gaze Tracking API input
    gaze_input_source = GAZE_INPUT_VIDEO_PATH if not USE_WEBCAM_FOR_GAZE else WEBCAM_ID
    gaze_input_cap, GAZE_CAM_WIDTH, GAZE_CAM_HEIGHT, GAZE_INPUT_FPS = initialize_video_source(
        gaze_input_source, is_file=(not USE_WEBCAM_FOR_GAZE)
    )
    if gaze_input_cap is None: sys.exit(1)

    # Source for Stimulus Display (only needed if running evaluation)
    stimulus_cap = None
    if RUN_ATTENTION_EVALUATION:
        stimulus_cap, _, _, _ = initialize_video_source(STIMULUS_VIDEO_PATH, is_file=True)
        # We can continue even if stimulus video fails, the evaluation step will just be skipped

    # --- Step 3: Initialize and Use API ---
    calibration_file_to_load = CALIBRATION_NPZ_FILENAME if LOAD_CALIBRATION else None
    calibration_successful = False
    api_instance = None # Define outside try for finally clause

    try:
        with GazeTrackerAPI(screen_width=SCREEN_WIDTH,
                            screen_height=SCREEN_HEIGHT,
                            stabilize_gaze=STABILIZE_GAZE,
                            calibration_file=calibration_file_to_load
                            ) as api_instance:

            print(f"\nAPI Initialized. Running: {api_instance.is_running}, Calibrated: {api_instance.is_calibrated}")
            calibration_needed = not api_instance.is_calibrated
            DEFAULT_PARAMS = api_instance.gaze_calibration.get_current_parameters()

            # --- Step 4: Run Calibration (if needed) ---
            if calibration_needed:
                if not USE_WEBCAM_FOR_GAZE: rewind_video_capture(gaze_input_cap, "gaze input video")
                calibration_successful = run_calibration_session(api_instance, gaze_input_cap, SCREEN_WIDTH, SCREEN_HEIGHT)
            else:
                print("Calibration loaded or not required.")
                calibration_successful = True

            # --- Step 5: Run Attention Evaluation ---
            if calibration_successful and RUN_ATTENTION_EVALUATION:
                if stimulus_cap is None:
                     print("WARN: Skipping Attention Evaluation as stimulus video failed to load.")
                else:
                    # Rewind sources if they are files before starting evaluation
                    if not USE_WEBCAM_FOR_GAZE: rewind_video_capture(gaze_input_cap, "gaze input video")
                    rewind_video_capture(stimulus_cap, "stimulus video")
                    run_attention_evaluation(api_instance, stimulus_cap, gaze_input_cap, SCREEN_WIDTH, SCREEN_HEIGHT)

            # --- Step 6: Run Simple Gaze Test ---
            if calibration_successful and RUN_SIMPLE_GAZE_TEST:
                 if not USE_WEBCAM_FOR_GAZE: rewind_video_capture(gaze_input_cap, "gaze input video")
                 run_simple_gaze_test_session(api_instance, gaze_input_cap, SCREEN_WIDTH, SCREEN_HEIGHT, GAZE_CAM_WIDTH, GAZE_CAM_HEIGHT, GAZE_INPUT_FPS)
            # (Skip messages based on flags/status)


    except Exception as e:
        print(f"\n--- An Error Occurred During API Usage ---")
        traceback.print_exc()

    finally:
        # --- Final Cleanup ---
        print("\n--- Cleaning Up Resources ---")
        if gaze_input_cap and gaze_input_cap.isOpened(): print("Releasing gaze input source..."); gaze_input_cap.release()
        if stimulus_cap and stimulus_cap.isOpened(): print("Releasing stimulus video source..."); stimulus_cap.release()
        print("Destroying any remaining OpenCV windows..."); cv2.destroyAllWindows()
        print("======= Gaze Tracker API Test Script Finished =======")
