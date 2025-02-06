import cv2
from modules.gaze_tracking.gaze_tracking import GazeTracking
from modules.gaze_tracking.gaze_calibration import GazeCalibration
from modules.gaze_tracking.point_of_gaze import PointOfGaze
from modules.gaze_tracking.screensize import get_screensize

def main():
    # Initialize GazeTracking
    gaze_tracking = GazeTracking()

    # Get the monitor screen size
    monitor = get_screensize()

    # Initialize GazeCalibration with the GazeTracking object and monitor information
    gaze_calibration = GazeCalibration(gaze_tracking, monitor)

    # Initialize PointOfGaze with the necessary parameters
    stabilize = True  # Set to True to enable gaze stabilization
    pog = PointOfGaze(gaze_tracking, gaze_calibration, monitor, stabilize)

    # Create a window for calibration and set it to fullscreen
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Open the webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get the frame width and height of the webcam to calculate the 'webcam_estate'
    frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    webcam_estate = frame_width * frame_height

    while True:
        # Capture frame from the webcam
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Refresh the gaze tracking analysis with the new frame
        gaze_tracking.refresh(frame)

        # If calibration is not complete, perform calibration
        if not gaze_calibration.is_completed():
            calib_frame = gaze_calibration.calibrate_gaze(webcam_estate)
            cv2.imshow('Calibration', calib_frame)  # Show the calibration dots
        elif not gaze_calibration.is_tested():  # If calibration is complete but testing is not
            test_frame = gaze_calibration.test_gaze(pog, webcam_estate)
            cv2.imshow('Calibration', test_frame)  # Show the test points and gaze estimation
        else:
            # Estimate the point of gaze on the screen after calibration and testing
            screen_x, screen_y = pog.point_of_gaze()
            print('start estimating')
            if screen_x is not None and screen_y is not None:
                print(f"Estimated gaze position on screen: ({screen_x}, {screen_y})")

        # Optionally display annotated frame with pupil positions marked
        # Uncomment these lines if you want to see the pupil tracking
        annotated_frame = gaze_tracking.annotated_frame()
        cv2.imshow("Gaze Tracking", annotated_frame)

        # Print gaze direction if pupils are located
        if gaze_tracking.pupils_located:
            print(f"Left pupil: {gaze_tracking.pupil_left_coords()}")
            print(f"Right pupil: {gaze_tracking.pupil_right_coords()}")
            print(f"Gaze is right: {gaze_tracking.is_right()}")
            print(f"Gaze is left: {gaze_tracking.is_left()}")
            print(f"Gaze is center: {gaze_tracking.is_center()}")
            print(f"Gaze is up: {gaze_tracking.is_up()}")
            print(f"Gaze is down: {gaze_tracking.is_down()}")

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close all windows
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()