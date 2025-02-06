import cv2
import os
import datetime
import AppKit
import Quartz
from platform import system
from modules.gaze_tracking.screensize import get_screensize
from modules.gaze_tracking.gaze_tracking import GazeTracking
from modules.gaze_tracking.gaze_calibration import GazeCalibration
from modules.gaze_tracking.point_of_gaze import PointOfGaze


class EPOGAnalyzer:
    def __init__(self, test_error_dir, argv):
        if len(argv) > 1 and argv[1] == '1':
            self.stabilize = True
        self.stabilize = True

        self.test_error_dir = test_error_dir
        self.test_error_file = self.setup_test_error_file(argv)

        self.webcam, self.webcam_w, self.webcam_h = self.setup_webcam()
        self.webcam_estate = self.webcam_w * self.webcam_h

        op_sys = system()
        if op_sys == 'Windows':
            self.monitor = get_screensize()  # dict: {width, height}
        elif op_sys == 'Darwin':
            screen = AppKit.NSScreen.mainScreen().visibleFrame()
            self.monitor = {'width': int(screen.size.width), 'height': int(screen.size.height)}
            print(f"Visible screen width: {int(screen.size.width)}, height: {int(screen.size.height)}")

        self.calib_window = self.setup_calib_window()  # string: window name
        self.windows_closed = False

        # Initialize gaze tracking using MediaPipe
        self.gaze_tr = GazeTracking()

        # Initialize gaze calibration and point of gaze estimation
        self.gaze_calib = GazeCalibration(self.gaze_tr, self.monitor, self.test_error_file)
        self.pog = PointOfGaze(self.gaze_tr, self.gaze_calib, self.monitor, self.stabilize)

    def setup_test_error_file(self, argv):
        test_error_file = None
        if len(argv) > 2:
            prefix = argv[2]
            if not os.path.isdir(self.test_error_dir):
                os.makedirs(self.test_error_dir)
            filename = os.path.join(self.test_error_dir, f"{prefix}_{'stab' if self.stabilize else 'raw'}_" +
                                    datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + '.txt')
            test_error_file = open(filename, 'w+')
        return test_error_file

    def setup_calib_window(self):
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration', self.monitor['width'], self.monitor['height'])
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)
        return 'Calibration'

    def setup_webcam(self):
        webcam = cv2.VideoCapture(0)
        webcam_w = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        webcam_h = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return webcam, webcam_w, webcam_h

    def perform_calibration(self):
        while True:
            _, frame = self.webcam.read()
            if frame is not None:
                screen_x, screen_y = self.analyze(frame)
                if screen_x is not None and screen_y is not None:
                    print(f"Calibration complete. Screen coordinates: {screen_x}, {screen_y}")
                    break
            if cv2.waitKey(1) == 27:
                self.webcam.release()
                cv2.destroyAllWindows()
                break

    def analyze(self, frame):
        self.gaze_tr.refresh(frame)  # Refresh and analyze frame
        screen_x, screen_y = None, None

        cv2.resizeWindow('Calibration', self.monitor['width'], self.monitor['height'])

        # Calibration process steps
        if not self.gaze_calib.is_completed():
            # Check if the window is out of screen bounds and reset if needed
            rect = cv2.getWindowImageRect(self.calib_window)

            # Ensure window position is within valid range
            valid_x = max(min(-rect[0], self.monitor['width'] - rect[2]), 0)
            valid_y = max(min(-rect[1], self.monitor['height'] - rect[3]), 0)

            cv2.moveWindow(self.calib_window, valid_x, valid_y)

            calib_frame = self.gaze_calib.calibrate_gaze(self.webcam_estate)
            cv2.imshow(self.calib_window, calib_frame)
        elif not self.gaze_calib.is_tested():
            calib_frame = self.gaze_calib.test_gaze(self.pog, self.webcam_estate)
            cv2.imshow(self.calib_window, calib_frame)
        else:
            if not self.windows_closed:
                icon_sz = 0
                cv2.resizeWindow(self.calib_window, icon_sz, icon_sz)
                cv2.moveWindow(self.calib_window, self.monitor['width'] - icon_sz, self.monitor['height'] - icon_sz)
                self.windows_closed = True
            screen_x, screen_y = self.pog.point_of_gaze()

        return screen_x, screen_y


if __name__ == "__main__":
    import sys
    test_error_dir = "path/to/test_error_dir"
    analyzer = EPOGAnalyzer(test_error_dir, sys.argv)
    analyzer.perform_calibration()