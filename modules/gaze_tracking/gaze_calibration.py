from __future__ import division

import math
import cv2
import numpy as np
from random import randint
import logging


class GazeCalibration(object):
    """
    This class calibrates the mapping of gaze to the screen size
    that the user is looking at.
    """

    def __init__(self, gaze_tracking, monitor, test_error_file=None):
        """
        :param gaze_tracking: object for gaze tracking that holds the detected values for eye and iris
        :param monitor: monitor size in pixels (w, h)
        :param test_error_file: file where epog test errors can be logged (can be None)
        """
        self.logger = logging.getLogger(__name__)

        self.gaze_tracking = gaze_tracking
        self.leftmost_hr = 0
        self.rightmost_hr = 0
        self.top_vr = 0
        self.bottom_vr = 0

        # Calibration points on the screen (odd number for center point calibration)
        self.nb_p = [3, 3]  # (vertical_nb_points, horizontal_nb_points)
        self.hr = []
        self.vr = []

        self.circle_rad = 20
        self.fs_frame = self.setup_calib_frame(monitor)  # Full screen calibration frame
        self.fsh, self.fsw = self.fs_frame.shape[:2]
        self.nb_calib_points, self.calib_points = self.setup_calib_points()
        self.nb_test_points = 5
        self.test_points = self.setup_test_points()

        self.calib_ratios = [[] for _ in range(self.nb_calib_points)]
        self.base_iris_size = 0
        self.iris_size_div = 0

        self.nb_instr_frames = 40  # Instruction frames for calibration
        self.nb_fixation_frames = 20  # Frames before measuring starts
        self.nb_calib_frames = 40  # Calibration frames after fixation
        self.nb_test_frames = 40  # Test frames for showing test points

        # Counters for calibration process
        self.instr_frame = 0
        self.calib_p = 0
        self.fixation_frame = 0
        self.calib_frame = 0
        self.test_p = 0
        self.test_frame = 0
        self.fl = 0

        self.test_error_file = test_error_file
        self.test_errors_dict = {'x': [], 'y': [], 'xy': []}
        self.calib_completed = False
        self.test_completed = False

    def setup_calib_points(self):
        """
        Prepares calibration points across the screen.
        """
        calib_points = []
        step_h = (self.fsw - 2 * self.circle_rad) // (self.nb_p[1] - 1)
        step_v = (self.fsh - 2 * self.circle_rad) // (self.nb_p[0] - 1)
        for v in range(self.nb_p[0]):
            for h in range(self.nb_p[1]):
                x = h * step_h + self.circle_rad
                y = v * step_v + self.circle_rad
                calib_points.append((x, y))
        return len(calib_points), calib_points

    def setup_test_points(self):
        """
        Sets up random test points on the screen.
        """
        test_points = []
        minx = self.circle_rad
        maxx = self.fsw - self.circle_rad
        miny = self.circle_rad
        maxy = self.fsh - self.circle_rad
        for _ in range(self.nb_test_points):
            test_points.append(np.asarray([randint(minx, maxx), randint(miny, maxy)]))
        return np.asarray(test_points)

    def setup_calib_frame(self, monitor):
        """
        Sets up the full-screen calibration frame.
        """
        self.logger.debug('Monitor resolution {} x {}'.format(monitor['height'], monitor['width']))
        fullscreen_frame = np.zeros((monitor['height'], monitor['width'], 3), np.uint8)
        return fullscreen_frame

    def calibrate_gaze(self, webcam_estate):
        """
        Display a fixation circle at each calibration point.
        """
        self.fs_frame.fill(50)
        if self.calib_p < self.nb_calib_points:
            if self.instr_frame < self.nb_instr_frames:
                self.display_instruction()
                self.instr_frame += 1
            elif self.fixation_frame < self.nb_fixation_frames:
                self.prompt_fixation(self.calib_p)
                self.fixation_frame += 1
            elif self.calib_frame < self.nb_calib_frames:
                self.prompt_fixation(self.calib_p)
                self.record_gaze_and_iris(self.calib_p, webcam_estate)
                self.calib_frame += 1
            else:
                self.calib_ratios[self.calib_p] = cluster_ratios_for_calib_point(self.calib_ratios[self.calib_p])
                self.calib_p += 1
                self.fl = self.calib_p
                self.fixation_frame = 0
                self.calib_frame = 0
        else:
            self.compute_extreme_ratios()
            if self.iris_size_div > 0:
                self.base_iris_size /= self.iris_size_div
            self.calib_completed = True
        return self.fs_frame

    def compute_extreme_ratios(self):
        """
        Compute the extreme ratios for calibration.
        """
        self.logger.debug('Clustered ratios: ' + ' '.join(map(str, self.calib_ratios)))
        vert_nb_p = self.nb_p[0]
        hor_nb_p = self.nb_p[1]
        leftmost_hrs = []
        rightmost_hrs = []
        top_vrs = []
        bottom_vrs = []

        for v in range(vert_nb_p):
            for h in range(hor_nb_p):
                i = v * hor_nb_p + h
                if self.calib_ratios[i] is not None:
                    if h == 0:
                        leftmost_hrs.append(self.calib_ratios[i][0])
                    elif h == hor_nb_p - 1:
                        rightmost_hrs.append(self.calib_ratios[i][0])
                    if v == 0:
                        top_vrs.append(self.calib_ratios[i][1])
                    elif v == vert_nb_p - 1:
                        bottom_vrs.append(self.calib_ratios[i][1])

        self.leftmost_hr = density_based_1d_cluster(leftmost_hrs)
        self.rightmost_hr = density_based_1d_cluster(rightmost_hrs)
        self.top_vr = density_based_1d_cluster(top_vrs)
        self.bottom_vr = density_based_1d_cluster(bottom_vrs)
        self.logger.debug('Extreme ratios: left {} right {} top {} bottom {}'
                          .format(self.leftmost_hr, self.rightmost_hr, self.top_vr, self.bottom_vr))

    def display_instruction(self):
        """
        Display instructions for calibration.
        """
        cv2.putText(self.fs_frame, 'Focus on the red dots', (80, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.fs_frame, 'Click the window to proceed', (80, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)

    def prompt_fixation(self, calib_p):
        """
        Display a fixation point during calibration.
        """
        cv2.circle(self.fs_frame, self.calib_points[calib_p], self.circle_rad, (0, 0, 255), -1)

    def record_gaze_and_iris(self, calib_p, webcam_estate):
        """
        Record gaze ratios and iris size for calibration.
        """
        hr = self.gaze_tracking.horizontal_ratio()
        vr = self.gaze_tracking.vertical_ratio()
        if self.fl == calib_p:
            print('\ncalib p', calib_p)
            print('hr', hr)
            print('vr', vr)
            self.fl = -1

        if hr is not None and vr is not None:
            self.calib_ratios[calib_p].append([hr, vr])

        if self.calib_points[calib_p][0] == self.fsw // 2 and self.calib_points[calib_p][1] == self.fsh // 2:
            iris_diam = self.measure_iris_diameter()
            self.base_iris_size += iris_diam
            self.iris_size_div += 1

    def test_gaze(self, pog, webcam_estate):
        """
        Displays test points (red circle) and estimated gaze (lightgrey smaller dots) on
        the screen (in the frame). This method tests the accuracy of the gaze estimation.

        :param pog: PointOfGaze object used for gaze estimation
        :param webcam_estate: The size of the webcam estate (frame size)
        :return: Frame with test points and gaze estimation displayed
        """
        self.fs_frame.fill(50)  # Fill frame with grey for better visibility
        if self.test_p < self.nb_test_points:
            if self.test_frame == 1:
                self.logger.debug('Test point {}'.format(self.test_points[self.test_p]))

            # Display test point during nb_test_frames
            if self.test_frame < self.nb_test_frames:
                # Draw a red test circle at the test point
                cv2.circle(self.fs_frame, self.test_points[self.test_p], self.circle_rad, (0, 0, 255), -1)

                # Estimate the gaze point on the screen using the PointOfGaze object
                est_x, est_y = pog.point_of_gaze()

                if est_x is not None and est_y is not None:
                    # Draw a small light grey dot where the gaze is estimated on the screen
                    cv2.circle(self.fs_frame, (est_x, est_y), self.circle_rad // 4, (170, 170, 170), -1)

                    # Calculate and log error between estimated point and test point
                    err_x = est_x - self.test_points[self.test_p, 0]
                    err_y = est_y - self.test_points[self.test_p, 1]
                    err_xy = np.linalg.norm(np.asarray([est_x, est_y] - self.test_points[self.test_p]))
                    self.test_errors_dict['x'].append(err_x)
                    self.test_errors_dict['y'].append(err_y)
                    self.test_errors_dict['xy'].append(err_xy)

                    if self.test_error_file is not None:
                        self.test_error_file.write("%f\n" % err_xy)

                    # if self.test_frame == 1:
                    #     self.logger.debug('Test point {} Estimated gaze {} Error {}'
                    #                       .format(self.test_points[self.test_p], (est_x, est_y), int(round(err))))

                self.test_frame += 1
            else:
                self.test_p += 1
                self.test_frame = 0
        else:
            if self.test_error_file is not None:
                self.test_error_file.close()
            self.test_completed = True

            print(f"Aggregated estimation errors\n \
                    \nX: {np.mean(np.abs(self.test_errors_dict['x']))} \
                    \nY: {np.mean(np.abs(self.test_errors_dict['y']))} \
                    \nXY: {np.mean(np.abs(self.test_errors_dict['xy']))}\n")
            # print(f"x:{self.test_errors_dict['x']} \
            #         \ny:{self.test_errors_dict['y']} \
            #         \nxy:{self.test_errors_dict['xy']}")

        return self.fs_frame

    def measure_iris_diameter(self):
        """
        Measures the iris diameter based on MediaPipe landmarks.

        :return: Returns the iris diameter based on the landmarks provided by MediaPipe.
        """
        if self.gaze_tracking.eye_right is not None:
            # Using MediaPipe landmarks to calculate the iris diameter
            right_eye_landmarks = self.gaze_tracking.eye_right.landmarks

            # Indices of the right eye iris landmarks in MediaPipe
            right_iris_landmarks = [474, 475, 476,
                                    477]  # Adjust these indices based on the actual indices provided by MediaPipe

            # Calculate the Euclidean distance between the iris landmarks to get the diameter
            iris_width = np.linalg.norm(np.array([
                right_eye_landmarks[right_iris_landmarks[0]].x - right_eye_landmarks[right_iris_landmarks[2]].x,
                right_eye_landmarks[right_iris_landmarks[0]].y - right_eye_landmarks[right_iris_landmarks[2]].y
            ]))
            iris_height = np.linalg.norm(np.array([
                right_eye_landmarks[right_iris_landmarks[1]].x - right_eye_landmarks[right_iris_landmarks[3]].x,
                right_eye_landmarks[right_iris_landmarks[1]].y - right_eye_landmarks[right_iris_landmarks[3]].y
            ]))

            iris_diameter_right = (iris_width + iris_height) / 2
        else:
            iris_diameter_right = None

        if self.gaze_tracking.eye_left is not None:
            # Using MediaPipe landmarks to calculate the iris diameter
            left_eye_landmarks = self.gaze_tracking.eye_left.landmarks

            # Indices of the left eye iris landmarks in MediaPipe
            left_iris_landmarks = [469, 470, 471,
                                   472]  # Adjust these indices based on the actual indices provided by MediaPipe

            # Calculate the Euclidean distance between the iris landmarks to get the diameter
            iris_width = np.linalg.norm(np.array([
                left_eye_landmarks[left_iris_landmarks[0]].x - left_eye_landmarks[left_iris_landmarks[2]].x,
                left_eye_landmarks[left_iris_landmarks[0]].y - left_eye_landmarks[left_iris_landmarks[2]].y
            ]))
            iris_height = np.linalg.norm(np.array([
                left_eye_landmarks[left_iris_landmarks[1]].x - left_eye_landmarks[left_iris_landmarks[3]].x,
                left_eye_landmarks[left_iris_landmarks[1]].y - left_eye_landmarks[left_iris_landmarks[3]].y
            ]))

            iris_diameter_left = (iris_width + iris_height) / 2
        else:
            iris_diameter_left = None

        # Average the diameters of both eyes if available
        if iris_diameter_right is None:
            if iris_diameter_left is None:
                return None  # No iris detected
            else:
                return iris_diameter_left * 2
        elif iris_diameter_left is None:
            return iris_diameter_right * 2
        else:
            return (iris_diameter_right + iris_diameter_left) / 2

    @staticmethod
    def calc_error(p1, p2):
        dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        return dist

    def is_completed(self):
        return self.calib_completed

    def is_tested(self):
        return self.test_completed


def cluster_ratios_for_calib_point(ratios):
    """
    Cluster the ratios obtained from the given calibration point.
    """
    hrs = np.zeros(len(ratios), np.double)
    vrs = np.zeros(len(ratios), np.double)
    i = 0
    for [hr, vr] in ratios:
        hrs[i] = hr
        vrs[i] = vr
        i += 1
    best_hr = density_based_1d_cluster(hrs)
    best_vr = density_based_1d_cluster(vrs)
    return [best_hr, best_vr]


def density_based_1d_cluster(data):
    """
    Function for determining the best value from a series of noisy measurements.
    """
    hist, bins = np.histogram(data, bins='auto')
    ix = np.argmax(hist)
    best_val = (bins[ix] + bins[ix + 1]) / 2
    return best_val