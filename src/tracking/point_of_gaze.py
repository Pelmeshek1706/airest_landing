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
        self.monitor = monitor

        # split initialization into meaningful subfunctions
        self._init_cluster_parameters()
        self._init_cluster_storage()
        self._init_movement_flags()
        # note: iris size is set externally from calibration

    def _init_cluster_parameters(self):
        """
        initialize cluster-related parameters.
        """
        self.nb_same = 2  # required consecutive moves for consistent movement
        self.nb_interv = 3  # allowed intervening inconsistencies
        self.cluster_min_size = 2  # minimum cluster size to be considered stable
        self.cluster_max_size = 20  # maximum cluster storage size
        # maximum allowed variation within a cluster (20% of screen width)
        self.max_intra_cluster_dist_x = 0.2 * self.monitor['width']
        self.max_intra_cluster_dist_y = 0.2 * self.monitor['height']

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

        self._update_iris_on_centered_gaze(raw_x, raw_y)
        return raw_x, raw_y

    def _update_iris_on_centered_gaze(self, raw_x, raw_y):
        """
        refresh the stored iris measurement when the gaze is centered.
        this is based on the assumption that the iris diameter is most accurately measured
        when the user is looking straight ahead.
        """
        if self.looking_straight_ahead(raw_x, raw_y, self.gaze_calibration):
            self.current_iris_size = self.gaze_calibration.measure_iris_diameter()
            self.logger.debug(f'iris updated to {self.current_iris_size}')

    def _compute_raw_gaze(self):
        if not self.gaze_tracking.pupils_located:
            print('Pupils are not located')
            return None

        if getattr(self, 'current_iris_size', None) in [None, 0]:
            print('current iris size not valid, setting to baseline value')
            self.current_iris_size = self.gaze_calibration.base_iris_size
            self.logger.debug(f'iris set to baseline {self.current_iris_size}')

        hr = self.gaze_tracking.horizontal_ratio()
        vr = self.gaze_tracking.vertical_ratio()

        raw_x = (self.gaze_calibration.poly_x[0] * hr**2 +
                self.gaze_calibration.poly_x[1] * vr**2 +
                self.gaze_calibration.poly_x[2] * hr * vr +
                self.gaze_calibration.poly_x[3] * hr +
                self.gaze_calibration.poly_x[4] * vr +
                self.gaze_calibration.poly_x[5])
        
        raw_y = (self.gaze_calibration.poly_y[0] * hr**2 +
                self.gaze_calibration.poly_y[1] * vr**2 +
                self.gaze_calibration.poly_y[2] * hr * vr +
                self.gaze_calibration.poly_y[3] * hr +
                self.gaze_calibration.poly_y[4] * vr +
                self.gaze_calibration.poly_y[5])

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
        :return: True if consistent movement is detected, False otherwise.
        """
        if not (self.candidate_eye_move or self.ongoing_eye_move):
            return False

        self.move_cluster_x.append(x)
        self.move_cluster_y.append(y)
        combined = list(self.current_cluster_x) + list(self.move_cluster_x)
        consistent = self._has_consistent_movement(combined, self.nb_same)

        # reset clusters regardless of consistency
        self._reset_all_clusters()
        self.ongoing_eye_move = consistent
        if consistent:
            self.candidate_eye_move = False

        return consistent

    def _update_candidate_cluster(self, x, y):
        """
        update the candidate cluster if the new point fits.
        
        :param x: current x coordinate.
        :param y: current y coordinate.
        :return: True if the candidate cluster was updated, False otherwise.
        """
        # if there's no candidate cluster data, nothing to update
        if not self.candidate_cluster_x:
            return False

        # if the new point does not fit within the candidate cluster,
        # clear the candidate data and return False immediately
        if not self._point_within_cluster(self.candidate_cluster_x, self.candidate_cluster_y, x, y):
            self.candidate_cluster_x.clear()
            self.candidate_cluster_y.clear()
            return False

        # if the point fits, append the new coordinates
        self.candidate_cluster_x.append(x)
        self.candidate_cluster_y.append(y)

        # if the candidate cluster has reached the minimum size,
        # promote it to the current cluster and clear the candidate cluster
        if len(self.candidate_cluster_x) >= self.cluster_min_size:
            self.current_cluster_x = self.candidate_cluster_x.copy()
            self.current_cluster_y = self.candidate_cluster_y.copy()
            self.candidate_cluster_x.clear()
            self.candidate_cluster_y.clear()

        return True

    def _update_current_cluster(self, x, y):
        """
        update the current cluster with the new point if it fits;
        otherwise, start a new candidate cluster.
        
        :param x: current x coordinate.
        :param y: current y coordinate.
        """
        if self._point_within_cluster(self.current_cluster_x, self.current_cluster_y, x, y):
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

    def _point_within_cluster(self, cluster_x, cluster_y, x, y):
        """
        check if the new point (x, y) is within the allowed normalized distance
        from all points in the cluster using the combined normalized metric.
        
        :param cluster_x: deque of x coordinates.
        :param cluster_y: deque of y coordinates.
        :param x: new x coordinate.
        :param y: new y coordinate.
        :return: True if within allowed distance, False otherwise.
        """
        if not cluster_x or not cluster_y:
            return True
        # combine the x and y values into (x,y) pairs
        cluster_points = list(zip(cluster_x, cluster_y))
        return self._within_cluster(cluster_points, x, y)
    
    def _within_cluster(self, cluster_points, x, y):
        """
        Check if the new point (x, y) is within the allowed normalized distance
        from all points in the cluster.
        
        :param cluster_points: list of (x, y) pairs.
        :param x: new x coordinate.
        :param y: new y coordinate.
        :return: True if the normalized Euclidean distance for each point is within threshold.
        """
        norm_threshold = 0.75

        cluster_x, cluster_y = [p[0] for p in cluster_points], [p[1] for p in cluster_points]
        mean_x = sum(cluster_x) / len(cluster_x)
        mean_y = sum(cluster_y) / len(cluster_y)
        dx = abs(mean_x - x) / self.monitor['width']
        dy = abs(mean_y - y) / self.monitor['height']
        norm_diff = np.sqrt(dx**2 + dy**2)
        return norm_diff <= norm_threshold  # use your chosen threshold

        # alpha, beta = 1.0, 1.0
        # for cx, cy in cluster_points:
        #     dx = abs(cx - x) / self.monitor['width']
        #     dy = abs(cy - y) / self.monitor['height']
        #     if np.sqrt(alpha * dx**2 + beta * dy**2) > norm_threshold:
        #         return False
        # return True

        # for cx, cy in zip(cluster_x, cluster_y):
        #     if abs(cx - x) > self.max_intra_cluster_dist_x or abs(cy - y) > self.max_intra_cluster_dist_y:
        #         return False
        # return True

    def _has_consistent_movement(self, values, nb_same):
        """
        check whether the last nb_same+1 values show consistent monotonic movement.
        allow up to self.nb_interv inconsistencies.
        
        :param values: list of numeric values.
        :param nb_same: required consecutive moves.
        :return: True if movement is consistent, False otherwise.
        """
        if len(values) < nb_same + 1:
            return False

        # count the number of inconsistencies over the last nb_same+1 values
        inconsistencies = sum(
            1
            for i in range(-nb_same - 1, -1)
            if (values[i+1] - values[i]) * (values[i+2] - values[i+1]) <= 0
        )
        return inconsistencies <= self.nb_interv

    @staticmethod
    def looking_straight_ahead(est_x, est_y, gaze_calib):
        """
        check if the estimated gaze is in the center region of the screen.
        
        :param est_x: estimated x coordinate.
        :param est_y: estimated y coordinate.
        :param gaze_calib: calibration object containing screen dimensions.
        :return: true if gaze is in the center region, false otherwise.
        """
        wmargin = gaze_calib.frame_width * 0.3
        hmargin = gaze_calib.frame_height * 0.5
        wmiddle = gaze_calib.frame_width / 2
        hmiddle = gaze_calib.frame_height / 2
        return (wmiddle - wmargin < est_x < wmiddle + wmargin) and (hmiddle - hmargin < est_y < hmiddle + hmargin)
    