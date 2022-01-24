import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of freedom (contains values for N=1, ..., 9). 
Taken from MATLAB/Octave's chi2inv function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
    }

class KalmanFilter(object):
    """A simple Kalman filter for tracking bounding boxes in image space
    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh) contains: 
        the bounding box center position (x, y), 
        aspect ratio a, height h,
        and their respective velocities va, vh.
    
    Object motion follows a constant velocity model. 
    The bounding box location (x, y, a, h) is taken as direct observation 
    of the state space (linear observation model).
    """
    def __init__(self):
        ndim, dt = 4, 1

        # create Kalman filter model matrics
        self._motion_mat = np.eye(ndim * 2, ndim * 2)
        for i in range(ndim):
            self._motion_mat[i, ndim+i] = dt
        
        self._update_mat = np.eye(ndim, ndim * 2)

        # motion and observation uncertainty are chosen relative to the current state estimate
        # these weights control the amount of uncertainity in the model
        self._std_weight_pos = 1. / 20
        self._std_weight_vel = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement
        Params:
            measurement(ndarray): bounding box coordinates (x, y, a, h)
        Returns:
            (ndarray, ndarray): the mean vector (8 dimensional) and covariance matrix 
            (8x8 dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_pos * measurement[0],  # center point x
            2 * self._std_weight_pos * measurement[1],  # center point y
            1 * measurement[2],                         # ratio of width/height
            2 * self._std_weight_pos * measurement[3],  # height
            10 * self._std_weight_vel * measurement[0],
            10 * self._std_weight_vel * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_vel * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction
        Params:
            mean(ndarray): 8 dimensional mean vector of the object state 
                at the previous time step.
            covariance(ndarray): 8x8 dimensional covariance matrix of the object state 
                at the previous time step.
        Returns:
            (ndarray, ndarray): the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_pos * mean[0],
            self._std_weight_pos * mean[1],
            1 * mean[2],
            self._std_weight_pos * mean[3],
        ]
        std_vel = [
            self._std_weight_vel * mean[0],
            self._std_weight_vel * mean[1],
            0.1 * mean[2],
            self._std_weight_vel * mean[3],
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space
        Params:
            mean(ndarray): the state's mean vector (8 dimensional array)
            covariance(ndarray): the state's covariance matrix (8x8 dimensional)
        Returns:
            (ndarray, ndarray): the projected mean and covariance matrix of the given state
            estimate
        """
        std = [
            self._std_weight_pos * mean[0],
            self._std_weight_pos * mean[1],
            0.1 * mean[2],
            self._std_weight_pos * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step
        Params:
            mean(ndarray): the predicted state's mean vector (8 dimensional).
            covariance(ndarray): the state's covariance matrix (8x8 dimensional).
            measurement(ndarray): the 4 dimensional measurement vector (x, y, a, h), 
                where (x, y) is the center position, a the aspect ratio, 
                and h the height of the bounding box.
        Returns:
            (ndarray, ndarray): returns the measurement-corrected state distribution.
        """
        proj_mean, proj_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), 
            np.dot(covariance, self._update_mat.T).T, 
            check_finite=False).T
        innovation = measurement - proj_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, proj_cov, kalman_gain.T))
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements
        A suitable distance threshold can be obtained from `chi2inv95`. 
        If `only_position` is False, the chi-square distribution has 4 degrees of freedom, otherwise 2.
        Params:
            mean(ndarray): mean vector over the state distribution (8 dimensional)
            covariance(ndarray): covariance of the state distribution (8x8 dimensional)
            measurements(ndarray): an Nx4 dimensional matrix of N measurements, 
                each in format (x, y, a, h) where (x, y) is the bounding box center position, 
                a the aspect ratio, and h the height.
            only_position(optional[bool]): if True, distance computation is done with respect 
                to the bounding box center position only.
        Returns:
            ndarray: returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        chol_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(chol_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha