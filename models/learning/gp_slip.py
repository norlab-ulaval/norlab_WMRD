import numpy as np
import pickle
from util.transform_algebra import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class FullBodySlipGaussianProcess:
    def __init__(self, n_dimensions_x, n_dimensions_y, n_dimensions_yaw, length_scales_x, length_scales_y, length_scales_yaw,
                 noise_levels_x, noise_levels_y, noise_levels_yaw, n_restarts_optimizer, baseline, radius, dt, kappa_param):

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
        self.next_sigma_states = np.zeros((3, self.n_sigma_points))
        self.sigma_sum = np.zeros((self.n_state_dimensions, self.n_state_dimensions))

        self.kernel_x = 1.0 * RBF(length_scale= length_scales_x) + WhiteKernel(noise_level=noise_levels_x)
        self.kernel_y = 1.0 * RBF(length_scale= length_scales_y) + WhiteKernel(noise_level=noise_levels_y)
        self.kernel_yaw = 1.0 * RBF(length_scale= length_scales_yaw) + WhiteKernel(noise_level=noise_levels_yaw)
        self.n_restarts_optimizer = n_restarts_optimizer

    def train_params(self, idd_velocities_x, idd_velocities_yaw, slip_velocities_x, slip_velocities_y, slip_velocities_yaw, print_score_bool):
        training_input_x = np.mean(idd_velocities_x, axis=1).reshape(-1, 1) # longitudinal body vel
        training_output_x = np.mean(slip_velocities_x, axis=1).reshape(-1, 1)  # longitudinal slip vel
        training_output_x_std = np.std(slip_velocities_x, axis=1)

        training_input_y = (np.mean(idd_velocities_x, axis=1) * np.mean(idd_velocities_yaw, axis=1)).reshape(-1, 1)   # centrifugal force
        training_output_y = np.mean(slip_velocities_y, axis=1).reshape(-1, 1)   # lateral slip vel
        training_output_y_std = np.std(slip_velocities_y, axis=1)


        training_input_yaw = np.column_stack((np.mean(idd_velocities_x, axis=1) * np.mean(idd_velocities_yaw, axis=1),  # centrifugal force
                                              np.mean(idd_velocities_x, axis=1), # assymetry
                                              np.mean(idd_velocities_yaw, axis=1))) # angular body vel
        training_output_yaw = np.mean(slip_velocities_yaw, axis=1).reshape(-1, 1)   # angular slip vel
        training_output_yaw_std = np.std(slip_velocities_yaw, axis=1)

        self.gaussian_process_slip_x = GaussianProcessRegressor(kernel=self.kernel_x,
                                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                                alpha=training_output_x_std**2).fit(training_input_x, training_output_x)
        self.gaussian_process_slip_y = GaussianProcessRegressor(kernel=self.kernel_y,
                                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                                alpha=training_output_y_std ** 2).fit(training_input_y,
                                                                                                      training_output_y)
        self.gaussian_process_slip_yaw = GaussianProcessRegressor(kernel=self.kernel_yaw,
                                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                                alpha=training_output_yaw_std ** 2).fit(training_input_yaw,
                                                                                                      training_output_yaw)

        if print_score_bool:
            print('gpr_x')
            print(self.gaussian_process_slip_x.score(training_input_x, training_output_x))
            print(self.gaussian_process_slip_x.kernel_)

            print('gpr_y')
            print(self.gaussian_process_slip_y.score(training_input_y, training_output_y))
            print(self.gaussian_process_slip_y.kernel_)

            print('gpr_yaw')
            print(self.gaussian_process_slip_yaw.score(training_input_yaw, training_output_yaw))
            print(self.gaussian_process_slip_yaw.kernel_)

    def save_params(self, param_directory_path):
        pkl_filename = param_directory_path + 'gaussian_process_slip_x.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.gaussian_process_slip_x, file)

        pkl_filename = param_directory_path + 'gaussian_process_slip_y.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.gaussian_process_slip_y, file)

        pkl_filename = param_directory_path + 'gaussian_process_slip_yaw.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.gaussian_process_slip_yaw, file)

    def load_params(self, param_directory_path):
        pkl_filename = param_directory_path + 'gaussian_process_slip_x.pkl'
        with open(pkl_filename, 'rb') as file:
            self.gaussian_process_slip_x = pickle.load(file)

        pkl_filename = param_directory_path + 'gaussian_process_slip_y.pkl'
        with open(pkl_filename, 'rb') as file:
            self.gaussian_process_slip_y = pickle.load(file)

        pkl_filename = param_directory_path + 'gaussian_process_slip_yaw.pkl'
        with open(pkl_filename, 'rb') as file:
            self.gaussian_process_slip_yaw = pickle.load(file)

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
    def predict_from_sigma_points(self, idd_vel_x, idd_vel_y, idd_vel_yaw):
        blr_body_to_world_rotmat = np.eye(2)
        n_sigma_points = self.sigma_points_array.shape[1]

        for i in range(0, n_sigma_points):
            blr_body_vel = np.array([idd_vel_x - (self.sigma_points_array[3, i]), idd_vel_y - (self.sigma_points_array[4, i])]).reshape(2, 1)
            yaw_to_rotmat2d(blr_body_to_world_rotmat, self.sigma_points_array[2, i])
            blr_world_vel = blr_body_to_world_rotmat @ blr_body_vel
            self.next_sigma_states[0, i] = self.sigma_points_array[0, i] + (blr_world_vel[0]) * self.dt
            self.next_sigma_states[1, i] = self.sigma_points_array[1, i] + (blr_world_vel[1]) * self.dt
            self.next_sigma_states[2, i] = self.sigma_points_array[2, i] + (idd_vel_yaw - self.sigma_points_array[5, i]) * self.dt

    def extract_mean_covariance_from_sigma_points(self):
        sigma_factor = 1 / (2 * self.n_state_dimensions + self.kappa_param)
        self.sigma_mean = sigma_factor * (self.kappa_param * self.next_sigma_states[:, 0] + 0.5 * np.sum(self.next_sigma_states[:, 1:], axis=1))
        for i in range(1, 4 * self.n_state_dimensions):
            self.sigma_sum += (self.next_sigma_states[:, i] - self.sigma_mean).reshape(3, 1) @ (self.next_sigma_states[:, i] - self.sigma_mean).reshape(1, 3)
        self.sigma_cov = sigma_factor * (self.kappa_param * ((self.next_sigma_states[:, 0] - self.sigma_mean).reshape(3, 1) @
                                                   (self.next_sigma_states[:, 0] - self.sigma_mean).reshape(1, 3)) + 0.5 * (self.sigma_sum))
        self.sigma_sum = np.zeros((self.n_state_dimensions, self.n_state_dimensions))
        return self.sigma_mean, self.sigma_cov

    def predict_slip_from_body_vels(self, body_idd_vels):
        prediction_input_x = body_idd_vels[:, 0].reshape(-1, 1)  # longitudinal body vel
        prediction_input_y = (body_idd_vels[:, 0] * body_idd_vels[:, 2]).reshape(-1, 1)  # centrifugal force
        prediction_input_yaw = np.column_stack((body_idd_vels[:, 0] * body_idd_vels[:, 2],  # centrifugal force
                                              body_idd_vels[:, 0],  # assymetry
                                              body_idd_vels[:, 2]))  # angular body vel

        # predicted_slip_x_mean, predicted_slip_x_cov = self.body_x_slip_blr.predict_slip(prediction_input_x)
        predicted_slip_x_mean, predicted_slip_x_cov = self.gaussian_process_slip_x.predict(prediction_input_x, return_cov=True)
        predicted_slip_y_mean, predicted_slip_y_cov = self.gaussian_process_slip_y.predict(prediction_input_y, return_cov=True)
        predicted_slip_yaw_mean, predicted_slip_yaw_cov = self.gaussian_process_slip_yaw.predict(prediction_input_yaw, return_cov=True)
        # predicted_slip_y_mean, predicted_slip_y_cov = self.body_y_slip_blr.predict_slip(prediction_input_y)
        # predicted_slip_yaw_mean, predicted_slip_yaw_cov = self.body_yaw_slip_blr.predict_slip(prediction_input_yaw)
        return predicted_slip_x_mean.reshape(-1, 1), predicted_slip_x_cov, \
            predicted_slip_y_mean.reshape(-1, 1), predicted_slip_y_cov, \
            predicted_slip_yaw_mean.reshape(-1, 1), predicted_slip_yaw_cov

    def predict_horizon_from_body_idd_vels(self, body_idd_vels, init_state, init_state_covariance):
        horizon_len = body_idd_vels.shape[0]
        predicted_slip_x_mean, predicted_slip_x_cov, predicted_slip_y_mean, predicted_slip_y_cov, predicted_slip_yaw_mean, predicted_slip_yaw_cov = self.predict_slip_from_body_vels(body_idd_vels)

        prediction_means = np.zeros((self.n_state_dimensions, horizon_len))
        prediction_means[:3, 0] = init_state
        prediction_covariances = np.zeros((2 * self.n_state_dimensions, 2 * self.n_state_dimensions, horizon_len))
        prediction_covariances[:3, :3, 0] = np.eye(3) * init_state_covariance

        for j in range(0, horizon_len-1):
            # print(predicted_slip_x_cov.shape)
            # print(predicted_slip_x_mean[j])
            prediction_slip_mean_vector = np.concatenate((prediction_means[:, j], predicted_slip_x_mean[j],
                                                          predicted_slip_y_mean[j], predicted_slip_yaw_mean[j]))
            prediction_covariances[3:, 3:, j] = np.diag(np.array([predicted_slip_x_cov[j, j],
                                                                  predicted_slip_y_cov[j, j],
                                                                  predicted_slip_yaw_cov[j, j]]))
            self.compute_sigma_points(prediction_slip_mean_vector, prediction_covariances[:, :, j])
            self.predict_from_sigma_points(body_idd_vels[j, 0], body_idd_vels[j, 1], body_idd_vels[j, 2])
            prediction_means[:, j+1], prediction_covariances[:3, :3, j+1] = self.extract_mean_covariance_from_sigma_points()

        return prediction_means, prediction_covariances[:3, :3, :]


