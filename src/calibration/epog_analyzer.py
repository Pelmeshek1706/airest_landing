import cv2
import numpy as np
from utils.utils import get_screen_size
from tracking.gaze_tracking import GazeTracking
from tracking.point_of_gaze import PointOfGaze
from calibration.gaze_calibration import GazeCalibration


class EPOGAnalyzer:
    def __init__(self, stabilize=False, video_source=None, record=False):
        """
        initialize the epog analyzer.

        :param stabilize: boolean flag; if true, gaze estimation is stabilized.
        """
        self.stabilize = stabilize
        self.test_error_file = None  # placeholder; set this to an open file if needed

        # setup webcam and compute its pixel area
        self.video_source, self.record = video_source, record
        self.setup_webcam()

        # get monitor dimensions as a dict: {'width': ..., 'height': ...}
        self.monitor = get_screen_size()

        # setup calibration window
        self.calib_window = self.setup_calib_window()
        self.windows_closed = False

        # initialize gaze tracking using mediapipe
        self.gaze_tr = GazeTracking()

        # initialize calibration and point-of-gaze objects
        self.gaze_calib = GazeCalibration(self.gaze_tr, self.monitor, 
                                          self.video_source, self.record)
        self.pog = PointOfGaze(self.gaze_tr, self.gaze_calib, self.monitor, self.stabilize)

        if self.record:  # or use 'XVID', 'H264', etc.
            _fps = 30.0
            _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter("output_videos/test_webcam.mp4", _fourcc, _fps, (self.webcam_w, self.webcam_h))
        else:
            self.video_writer = None

    def setup_calib_window(self):
        """
        create and configure the calibration window.

        :return: window name.
        """
        window_name = 'calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.monitor['width'], self.monitor['height'])
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)
        return window_name

    def setup_webcam(self):
        """
        open the default webcam and get its dimensions.

        :return: tuple (webcam object, width, height)
        """
        if self.video_source:
            self.webcam = cv2.VideoCapture(self.video_source)
            self.record = False
        else:
            self.webcam = cv2.VideoCapture(0)

        self.webcam_w = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.webcam_h = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def perform_calibration(self):
        """
        main loop: capture frames and analyze them until calibration is complete
        or the escape key is pressed.
        """
        while True:
            ret, frame = self.webcam.read()
            if not ret or frame is None:
                continue
            
            # if recording is enabled, write the frame to file
            if self.video_writer:
                self.video_writer.write(frame)

            _, _ = self.analyze(frame)
                # if done with both calibration and testing, or ESC pressed, break
            if self.gaze_calib.is_completed() and self.gaze_calib.is_tested():
                break

            # check for escape key (27) -- mask low-order bits for compatibility
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.webcam.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def analyze(self, frame):
        """
        analyze a webcam frame by refreshing gaze tracking and updating the
        calibration or test phase.

        :param frame: image frame from the webcam.
        :return: tuple (screen_x, screen_y) if calibration and testing are complete;
                otherwise, (None, None)
        """
        # refresh and analyze the current frame for gaze data
        self.gaze_tr.refresh(frame)
        
        # always ensure the calibration window is sized to the monitor
        cv2.resizeWindow(self.calib_window, self.monitor['width'], self.monitor['height'])
        
        # if calibration is not complete, update the window position and show the calibration frame
        if not self.gaze_calib.is_completed():
            self._update_calib_window_position()
            calib_frame = self.gaze_calib.calibrate_gaze(self.pog)
            cv2.imshow(self.calib_window, calib_frame)
            return None, None
        
        # if calibration is complete but testing is not, show the test frame
        if not self.gaze_calib.is_tested():
            calib_frame = self.gaze_calib.test_gaze(self.pog)
            cv2.imshow(self.calib_window, calib_frame)
            return None, None
        
        # if both calibration and testing are complete, close the calibration window (if not already done)
        if not self.windows_closed:
            self._close_calib_window()
        
        # get the final gaze point
        return True, True

    def _update_calib_window_position(self):
        """
        check and update the calibration window position so that it stays
        within the monitor bounds.
        """
        rect = cv2.getWindowImageRect(self.calib_window)
        # calculate valid x and y positions for the window
        valid_x = max(min(-rect[0], self.monitor['width'] - rect[2]), 0)
        valid_y = max(min(-rect[1], self.monitor['height'] - rect[3]), 0)
        cv2.moveWindow(self.calib_window, valid_x, valid_y)

    def _close_calib_window(self):
        """
        hide the calibration window by resizing and moving it off-screen.
        """
        icon_sz = 0
        cv2.resizeWindow(self.calib_window, icon_sz, icon_sz)
        cv2.moveWindow(self.calib_window, self.monitor['width'] - icon_sz, self.monitor['height'] - icon_sz)
        self.windows_closed = True
