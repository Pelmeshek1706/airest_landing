from collections import deque
import logging
import numpy as np

class PointOfGaze:
    """
    class to track the user's gaze on the screen.
    provides estimated gaze coordinates on the computer screen.
    """

    def __init__(self, gaze_tracking, gaze_calibration, monitor, stabilize=False):
        """
        initialize the point-of-gaze object.

        :param gaze_tracking: gaze tracking object containing detected eye/iris info.
        :param gaze_calibration: calibration object with baseline iris size and extreme gaze ratios.
        :param monitor: dict with monitor dimensions, e.g. {'width': ..., 'height': ...}
        :param stabilize: boolean flag; if true, estimated gaze coordinates are stabilized.
        """
        self.logger = logging.getLogger(__name__)
        self.gaze_tracking = gaze_tracking
        self.gaze_calibration = gaze_calibration
        self.stabilize = stabilize

        # split initialization into meaningful subfunctions
        self._init_cluster_parameters(monitor)
        self._init_cluster_storage()
        self._init_movement_flags()
        # note: iris size is set externally from calibration

    def _init_cluster_parameters(self, monitor):
        """
        initialize cluster-related parameters.
        """
        self.nb_same = 2  # required consecutive moves for consistent movement
        self.nb_interv = 0  # allowed intervening inconsistencies
        self.cluster_min_size = 2  # minimum cluster size to be considered stable
        self.cluster_max_size = 20  # maximum cluster storage size
        # maximum allowed variation within a cluster (30% of screen width)
        self.max_intra_cluster_dist = 0.3 * monitor['width']

    def _init_cluster_storage(self):
        """
        initialize deques for storing gaze point clusters.
        """
        self.current_cluster_x = deque(maxlen=self.cluster_max_size)
        self.current_cluster_y = deque(maxlen=self.cluster_max_size)
        self.candidate_cluster_x = deque(maxlen=self.cluster_max_size)
        self.candidate_cluster_y = deque(maxlen=self.cluster_max_size)
        self.move_cluster_x = deque(maxlen=self.cluster_max_size)
        self.move_cluster_y = deque(maxlen=self.cluster_max_size)

    def _init_movement_flags(self):
        """
        initialize movement-related flags.
        """
        self.ongoing_eye_move = False
        self.candidate_eye_move = False

    def point_of_gaze(self):
        """
        compute raw gaze estimate and return a stabilized value if needed.

        :return: tuple (x, y) of estimated gaze coordinates, or (None, None) on failure.
        """
        raw = self._compute_raw_gaze()
        if raw is None:
            return (None, None)
        raw_x, raw_y = raw

        if self.stabilize:
            return self._stabilize_point(raw_x, raw_y)
        else:
            # optionally update iris measurement if gaze appears centered
            if self.looking_straight_ahead(raw_x, raw_y, self.gaze_calibration):
                self.current_iris_size = self.gaze_calibration.measure_iris_diameter()
                self.logger.debug(f'iris updated to {self.current_iris_size}')
            return raw_x, raw_y

    def _compute_raw_gaze(self):
        """
        compute raw gaze coordinates using horizontal and vertical ratios.
        returns (x, y) or None if gaze cannot be computed.
        """
        if not self.gaze_tracking.pupils_located:
            print('Pupils are not located')
            return None

        # if iris size is not set, use the calibration baseline
        if not hasattr(self, 'current_iris_size') or self.current_iris_size is None:
            print('Could not identify current iris size, setting to the baseline value')
            self.current_iris_size = self.gaze_calibration.base_iris_size
            self.logger.debug(f'iris set to baseline {self.current_iris_size}')

        dist_factor = self.gaze_calibration.base_iris_size / self.current_iris_size
        hr = self.gaze_tracking.horizontal_ratio()
        vr = self.gaze_tracking.vertical_ratio()

        try:
            raw_x = (max(self.gaze_calibration.leftmost_hr - hr, 0) *
                     self.gaze_calibration.frame_width * dist_factor) / \
                    (self.gaze_calibration.leftmost_hr - self.gaze_calibration.rightmost_hr)
            raw_y = (max(vr - self.gaze_calibration.top_vr, 0) *
                     self.gaze_calibration.frame_height * dist_factor) / \
                    (self.gaze_calibration.bottom_vr - self.gaze_calibration.top_vr)
            
        except ZeroDivisionError:
            print("division error in raw gaze computation")
            return None

        if np.isnan(raw_x) or np.isnan(raw_y):
            print('nan encountered in gaze estimation')
            return None

        return int(round(raw_x)), int(round(raw_y))

    def _stabilize_point(self, x, y):
        """
        stabilize the estimated gaze point by updating and using clusters.

        :param x: raw estimated x coordinate.
        :param y: raw estimated y coordinate.
        :return: stabilized (x, y) coordinates.
        """
        # part 1: handle movement mode if a movement is already flagged
        if self._handle_movement_mode(x, y):
            return x, y

        # part 2: update candidate cluster if it exists and fits the new point
        if self._update_candidate_cluster(x, y):
            return self._get_cluster_mean(x, y)

        # part 3: update current cluster; if the new point does not fit, start a candidate cluster
        self._update_current_cluster(x, y)
        return self._get_cluster_mean(x, y)

    def _handle_movement_mode(self, x, y):
        """
        if a candidate or ongoing movement is active, update the move clusters and check for consistency.
        
        :param x: current x coordinate.
        :param y: current y coordinate.
        :return: true if consistent movement is detected, false otherwise.
        """
        if self.candidate_eye_move or self.ongoing_eye_move:
            self.move_cluster_x.append(x)
            self.move_cluster_y.append(y)
            combined = list(self.current_cluster_x) + list(self.move_cluster_x)
            if self._has_consistent_movement(combined, self.nb_same):
                self.ongoing_eye_move = True
                self.candidate_eye_move = False
                self._reset_all_clusters()
                return True
            else:
                self.ongoing_eye_move = False
                self._reset_all_clusters()
        return False

    def _update_candidate_cluster(self, x, y):
        """
        update the candidate cluster if the new point fits.
        
        :param x: current x coordinate.
        :param y: current y coordinate.
        :return: true if the candidate cluster was updated, false otherwise.
        """
        if self.candidate_cluster_x:
            if self._point_within_cluster(self.candidate_cluster_x, x):
                self.candidate_cluster_x.append(x)
                self.candidate_cluster_y.append(y)
                # promote candidate cluster if it reaches minimum size
                if len(self.candidate_cluster_x) >= self.cluster_min_size:
                    self.current_cluster_x = self.candidate_cluster_x.copy()
                    self.current_cluster_y = self.candidate_cluster_y.copy()
                    self.candidate_cluster_x.clear()
                    self.candidate_cluster_y.clear()
                return True
            else:
                self.candidate_cluster_x.clear()
                self.candidate_cluster_y.clear()
        return False

    def _update_current_cluster(self, x, y):
        """
        update the current cluster with the new point if it fits;
        otherwise, start a new candidate cluster.
        
        :param x: current x coordinate.
        :param y: current y coordinate.
        """
        if self._point_within_cluster(self.current_cluster_x, x):
            self.current_cluster_x.append(x)
            self.current_cluster_y.append(y)
        else:
            self.candidate_eye_move = True
            self.move_cluster_x.append(x)
            self.move_cluster_y.append(y)
            self.candidate_cluster_x.append(x)
            self.candidate_cluster_y.append(y)

    def _get_cluster_mean(self, default_x, default_y):
        """
        return the mean (rounded) of the current cluster if available,
        otherwise the candidate cluster, or the default value.
        
        :param default_x: fallback x coordinate.
        :param default_y: fallback y coordinate.
        :return: tuple (mean_x, mean_y)
        """
        if self.current_cluster_x and self.current_cluster_y:
            mean_x = round(sum(self.current_cluster_x) / len(self.current_cluster_x))
            mean_y = round(sum(self.current_cluster_y) / len(self.current_cluster_y))
            return mean_x, mean_y
        elif self.candidate_cluster_x and self.candidate_cluster_y:
            mean_x = round(sum(self.candidate_cluster_x) / len(self.candidate_cluster_x))
            mean_y = round(sum(self.candidate_cluster_y) / len(self.candidate_cluster_y))
            return mean_x, mean_y
        else:
            return default_x, default_y

    def _reset_all_clusters(self):
        """
        reset all movement-related clusters.
        """
        self.current_cluster_x.clear()
        self.current_cluster_y.clear()
        self.candidate_cluster_x.clear()
        self.candidate_cluster_y.clear()
        self.move_cluster_x.clear()
        self.move_cluster_y.clear()

    def _point_within_cluster(self, cluster, x):
        """
        check if x is within the allowed distance of every element in cluster.
        
        :param cluster: deque of numeric values.
        :param x: value to check.
        :return: true if x is within the allowed distance, false otherwise.
        """
        for val in cluster:
            if abs(val - x) > self.max_intra_cluster_dist:
                return False
        return True

    def _has_consistent_movement(self, values, nb_same):
        """
        check whether the last nb_same+1 values show consistent monotonic movement.
        allow up to self.nb_interv inconsistencies.
        
        :param values: list of numeric values.
        :param nb_same: required consecutive moves.
        :return: true if movement is consistent, false otherwise.
        """
        if len(values) < nb_same + 1:
            return False
        allowed = self.nb_interv
        for i in range(-nb_same - 1, -1):
            diff1 = values[i+1] - values[i]
            diff2 = values[i+2] - values[i+1]
            if diff1 * diff2 <= 0:
                allowed -= 1
                if allowed < 0:
                    return False
        return True

    @staticmethod
    def looking_straight_ahead(est_x, est_y, gaze_calib):
        """
        check if the estimated gaze is in the center region of the screen.
        
        :param est_x: estimated x coordinate.
        :param est_y: estimated y coordinate.
        :param gaze_calib: calibration object containing screen dimensions.
        :return: true if gaze is in the center region, false otherwise.
        """
        wmargin = gaze_calib.fsw * 0.3
        hmargin = gaze_calib.fsh * 0.5
        wmiddle = gaze_calib.fsw / 2
        hmiddle = gaze_calib.fsh / 2
        return (wmiddle - wmargin < est_x < wmiddle + wmargin) and (hmiddle - hmargin < est_y < hmiddle + hmargin)
    