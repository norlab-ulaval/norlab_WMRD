import numpy as np
from scipy.stats import norm, invgamma, t

class SlipBayesianLinearRegression:
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

    def predict(self, x_data):
        n_data_points = x_data.shape[0]
        x_data_vector = x_data.reshape(n_data_points, self.n_dimensions)
        prediction_mean = x_data_vector @ self.weights
        variance_param = self.b_param_n / self.a_param_n * (np.eye(n_data_points) + x_data_vector @ self.params_covariance @ x_data_vector.T)
        scale_param = 2 * self.a_param_n
        prediction_variance = scale_param * variance_param / (scale_param - 2)

        return prediction_mean, prediction_variance
