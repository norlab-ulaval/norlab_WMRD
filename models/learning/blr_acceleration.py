import numpy as np
from util.transform_algebra import *

class AccelerationBayesianLinearRegression:
    def __init__(self, n_dimensions, a_param_init, b_param_init, param_variance_init, variance_init):
        self.n_dimensions = n_dimensions
        self.weights_init = np.zeros(n_dimensions)  # w_0
        self.a_param_init = a_param_init  # a_0
        self.b_param_init = b_param_init  # b_0
        self.param_variance_init = param_variance_init  # tau_0
        self.variance_init = variance_init  # sigma_0
        self.params_covariance_init = param_variance_init / variance_init * np.eye(n_dimensions)
        self.inv_params_covariance_init = np.linalg.inv(self.params_covariance_init)

        self.params_covariance = self.params_covariance_init
        self.inv_params_covariance = self.inv_params_covariance_init
        self.weights = self.weights_init
        self.a_param_n = self.a_param_init
        self.b_param_n = self.b_param_init
    def train_params(self, x_data, y_data):
        n_data = x_data.shape[0]
        self.inv_params_covariance = self.inv_params_covariance_init + x_data.T @ x_data
        self.params_covariance = np.linalg.inv(self.inv_params_covariance)
        self.weights = self.params_covariance @ (self.inv_params_covariance_init @ self.weights_init + x_data.T @ y_data)
        self.weights = self.weights.reshape(self.n_dimensions, 1)
        self.a_param_n = self.a_param_init + n_data / 2
        self.b_param_n = self.b_param_init + 0.5 * (self.weights_init.T @ self.inv_params_covariance_init @ self.weights_init +
                                               y_data.T @ y_data - self.weights.T @ self.inv_params_covariance @ self.weights)

    def predict_acceleration(self, x_data):
        n_data_points = x_data.shape[0]
        x_data_vector = x_data.reshape(n_data_points, self.n_dimensions)
        prediction_mean = x_data_vector @ self.weights
        variance_param = self.b_param_n / self.a_param_n * (np.eye(n_data_points) + x_data_vector @ self.params_covariance @ x_data_vector.T)
        scale_param = 2 * self.a_param_n
        prediction_variance = scale_param * variance_param / (scale_param - 2)

        return prediction_mean, prediction_variance

class FullBodyAccelerationBayesianLinearRegression:
    def __init__(self, n_dimensions_x, n_dimensions_yaw, a_param_init, b_param_init, param_variance_init, variance_init, baseline, radius, dt, kappa_param):

        self.body_x_acceleration_blr = AccelerationBayesianLinearRegression(n_dimensions_x, a_param_init, b_param_init, param_variance_init, variance_init)
        self.body_yaw_acceleration_blr = AccelerationBayesianLinearRegression(n_dimensions_yaw, a_param_init, b_param_init, param_variance_init, variance_init)

        self.jacobian = radius * np.array([[0.5, 0.5],
                                      [-1 / (baseline), 1 / (baseline)]])

        self.jacobian_3x3 = radius * np.array([[0.5, 0.5],
                                          [0.0, 0.0],
                                          [-1 / (baseline), 1 / (baseline)]])
        self.inv_jacobian = np.linalg.inv(self.jacobian)
        self.rotation_body_to_world = np.eye(2)
        self.body_vel_world_3d = np.zeros(6)
        self.body_vel_world_2d = np.zeros(3)
        self.state_2d = np.zeros(3)
        self.dt = dt

        self.kappa_param = kappa_param
        self.n_state_dimensions = 3
        self.n_sigma_points = 4 * self.n_state_dimensions + 1
        self.sigma_points_array = np.zeros((6, self.n_sigma_points))
        self.next_sigma_states = np.zeros((6, self.n_sigma_points))
        self.sigma_sum = np.zeros((self.n_state_dimensions, self.n_state_dimensions))

    def train_params(self, idd_velocities, icp_velocities, body_accelerations):
        training_input_x = np.column_stack((idd_velocities[:, 0],  # commanded_velocity
                                            icp_velocities[:, 0])) # measured_body_velocity
        training_input_yaw = np.column_stack((idd_velocities[:, 2],  # commanded_velocity
                                            icp_velocities[:, 2])) # measured_body_velocity

        self.body_x_acceleration_blr.train_params(training_input_x, body_accelerations[:, 0])
        self.body_yaw_acceleration_blr.train_params(training_input_yaw, body_accelerations[:, 2])

    def save_params(self, param_directory_path):
        np.save(param_directory_path + 'weights_x.npy', self.body_x_acceleration_blr.weights)
        np.save(param_directory_path + 'weights_yaw.npy', self.body_yaw_acceleration_blr.weights)

        np.save(param_directory_path + 'covariance_x.npy', self.body_x_acceleration_blr.params_covariance)
        np.save(param_directory_path + 'covariance_yaw.npy', self.body_yaw_acceleration_blr.params_covariance)

        np.save(param_directory_path + 'a_param_n_x.npy', self.body_x_acceleration_blr.a_param_n)
        np.save(param_directory_path + 'a_param_n_yaw.npy', self.body_yaw_acceleration_blr.a_param_n)

        np.save(param_directory_path + 'b_param_n_x.npy', self.body_x_acceleration_blr.b_param_n)
        np.save(param_directory_path + 'b_param_n_yaw.npy', self.body_yaw_acceleration_blr.b_param_n)

    def load_params(self, param_directory_path):
        self.body_x_acceleration_blr.weights = np.load(param_directory_path + 'weights_x.npy')
        self.body_yaw_acceleration_blr.weights = np.load(param_directory_path + 'weights_yaw.npy')

        self.body_x_acceleration_blr.params_covariance = np.load(param_directory_path + 'covariance_x.npy')
        self.body_yaw_acceleration_blr.params_covariance = np.load(param_directory_path + 'covariance_yaw.npy')

        self.body_x_acceleration_blr.inv_params_covariance = np.linalg.inv(self.body_x_acceleration_blr.params_covariance)
        self.body_yaw_acceleration_blr.inv_params_covariance = np.linalg.inv(self.body_yaw_acceleration_blr.params_covariance)

        self.body_x_acceleration_blr.a_param_n = np.load(param_directory_path + 'a_param_n_x.npy')
        self.body_yaw_acceleration_blr.a_param_n = np.load(param_directory_path + 'a_param_n_yaw.npy')

        self.body_x_acceleration_blr.b_param_n = np.load(param_directory_path + 'b_param_n_x.npy')
        self.body_yaw_acceleration_blr.b_param_n = np.load(param_directory_path + 'b_param_n_yaw.npy')

    def compute_body_vel(self, input):
        return self.jacobian @ input

    def compute_body_vel_horizon(self, horizon_input):
        return self.jacobian @ input

    def compute_sigma_points(self, mean_state_disturbance_vector, covariance):
        cholesky_matrix = np.linalg.cholesky(covariance)
        self.sigma_points_array[:, 0] = mean_state_disturbance_vector
        sigma_step = np.sqrt(2 * self.n_state_dimensions + self.kappa_param)
        for i in range(1, 2 * self.n_state_dimensions + 1):
            self.sigma_points_array[:, i] = mean_state_disturbance_vector + sigma_step * cholesky_matrix[:, i - 1]
            self.sigma_points_array[:, i + 2 * self.n_state_dimensions] = mean_state_disturbance_vector - sigma_step * cholesky_matrix[:, i - 1]
    def predict_from_sigma_points(self):
        blr_body_to_world_rotmat = np.eye(2)
        n_sigma_points = self.sigma_points_array.shape[1]

        for i in range(1, n_sigma_points):

            blr_body_vel = np.array([self.next_sigma_states[3, i-1], 0]).reshape(2, 1)
            yaw_to_rotmat2d(blr_body_to_world_rotmat, self.sigma_points_array[2, i])
            blr_world_vel = blr_body_to_world_rotmat @ blr_body_vel
            self.next_sigma_states[0, i] = self.next_sigma_states[0, i] + (blr_world_vel[0]) * self.dt
            self.next_sigma_states[1, i] = self.next_sigma_states[1, i-1] + (blr_world_vel[1]) * self.dt
            self.next_sigma_states[2, i] = self.next_sigma_states[2, i] + self.sigma_points_array[5, i-1] * self.dt
            self.next_sigma_states[3, i] = self.next_sigma_states[3, i] + self.sigma_points_array[3, i] * self.dt
            self.next_sigma_states[4, i] = 0.0
            self.next_sigma_states[5, i] = self.next_sigma_states[5, i] + self.sigma_points_array[5, i] * self.dt

    def extract_mean_covariance_from_sigma_points(self):
        sigma_factor = 1 / (2 * self.n_state_dimensions + self.kappa_param)
        self.sigma_mean = sigma_factor * (self.kappa_param * self.next_sigma_states[:, 0] + 0.5 * np.sum(self.next_sigma_states[:, 1:], axis=1))
        for i in range(1, 4 * self.n_state_dimensions):
            self.sigma_sum += (self.next_sigma_states[:, i] - self.sigma_mean).reshape(3, 1) @ (self.next_sigma_states[:, i] - self.sigma_mean).reshape(1, 3)
        self.sigma_cov = sigma_factor * (self.kappa_param * ((self.next_sigma_states[:, 0] - self.sigma_mean).reshape(3, 1) @
                                                   (self.next_sigma_states[:, 0] - self.sigma_mean).reshape(1, 3)) + 0.5 * (self.sigma_sum))
        self.sigma_sum = np.zeros((self.n_state_dimensions, self.n_state_dimensions))
        return self.sigma_mean, self.sigma_cov

    def predict_acceleration_from_body_vels(self, body_idd_vels, body_vels):
        prediction_input_x = np.column_stack((body_idd_vels[0],  # commanded_velocity
                                            body_vels[0]))  # predicted_body_velocity
        prediction_input_yaw = np.column_stack((body_idd_vels[2],  # commanded_velocity
                                              body_vels[2])) # predicted_body_velocity

        predicted_acceleration_x_mean, predicted_acceleration_x_cov = self.body_x_acceleration_blr.predict_acceleration(prediction_input_x)
        predicted_acceleration_y_mean = np.array([0.0]).reshape(1,1)
        predicted_acceleration_y_cov = 0.0
        predicted_acceleration_yaw_mean, predicted_acceleration_yaw_cov = self.body_yaw_acceleration_blr.predict_acceleration(prediction_input_yaw)
        return predicted_acceleration_x_mean, predicted_acceleration_x_cov, predicted_acceleration_y_mean, predicted_acceleration_y_cov, predicted_acceleration_yaw_mean, predicted_acceleration_yaw_cov

    def predict_horizon_from_body_idd_vels(self, body_idd_vels, init_state, init_state_covariance, init_body_vels):
        horizon_len = body_idd_vels.shape[0]
        body_vels = init_body_vels
        prediction_means = np.zeros((self.n_state_dimensions, horizon_len))
        prediction_means[:3, 0] = init_state
        prediction_covariances = np.zeros((2 * self.n_state_dimensions, 2 * self.n_state_dimensions, horizon_len))
        prediction_covariances[:3, :3, 0] = np.eye(3) * init_state_covariance

        for j in range(0, horizon_len-1):
            predicted_acceleration_x_mean, predicted_acceleration_x_cov, predicted_acceleration_y_mean, predicted_acceleration_y_cov, predicted_acceleration_yaw_mean, predicted_acceleration_yaw_cov = self.predict_acceleration_from_body_vels(body_idd_vels[j, :], body_vels)

            prediction_acceleration_mean_vector = np.concatenate((prediction_means[:, j].reshape(3,1), predicted_acceleration_x_mean,
                                                          predicted_acceleration_y_mean, predicted_acceleration_yaw_mean))
            prediction_covariances[3:, 3:, j] = np.diag(np.array([predicted_acceleration_x_cov,
                                                                  predicted_acceleration_y_cov,
                                                                  predicted_acceleration_yaw_cov]))
            self.compute_sigma_points(prediction_acceleration_mean_vector, prediction_covariances[:, :, j])
            self.predict_from_sigma_points(body_vels[0], body_vels[1], body_vels[2])
            body_vels[0] = body_vels[0] + predicted_acceleration_x_mean * self.dt
            body_vels[1] = body_vels[1] + predicted_acceleration_y_mean * self.dt
            body_vels[2] = body_vels[2] + predicted_acceleration_yaw_mean * self.dt
            prediction_means[:, j+1], prediction_covariances[:3, :3, j+1] = self.extract_mean_covariance_from_sigma_points()

        return prediction_means, prediction_covariances[:3, :3, :]


