import numpy as np
import pandas as pd
import torch

from eval.torch_dataset import TorchWMRDataset

from scipy.optimize import minimize

from util.util_func import *

class Stochastic_Trainer:
    def __init__(self, model, init_stoch_params, dataloader, timesteps_per_horizon, init_pred_covariance):
        self.model = model
        self.dataloader = dataloader
        self.timesteps_per_horizon = timesteps_per_horizon
        self.params = init_stoch_params
        self.init_pred_covariance = np.zeros((3,3))

    def update_dataloader(self, new_dataloader):
        self.dataloader = new_dataloader

    def update_stoch_params(self, new_stoch_params):
        self.model.adjust_stoch_params(new_stoch_params)

    def compute_stoch_error_all_steps(self, init_stoch_params):
        self.update_stoch_params(init_stoch_params)
        print(init_stoch_params)
        stochastic_prediction_error = 0
        counted_pred_counter = 0
        measurement_covariance = np.zeros((6,6))
        # self.x_train[idx], self.y_train[idx], self.calib_step[idx], self.mask[idx], self.cmd_vx[idx], self.cmd_omega[idx], \
        #                self.encoder_vx[idx], self.encoder_omega[idx], self.icp_vx[idx], self.icp_vy[idx], self.icp_omega[idx]
        for i, (inputs, targets, step, icp_vx, icp_vy, icp_omega, steady_state_mask, calib_mask) in enumerate(self.dataloader):
            # print(inputs)
            # print(targets)
            predicted_state = inputs[0, :6].numpy()
            predicted_covariance = self.init_pred_covariance
            steady_state_mask_bool = steady_state_mask.numpy()
            calib_mask_bool = calib_mask.numpy()
            if steady_state_mask_bool:
            # if True:
                for j in range(0, self.timesteps_per_horizon):
                    input_id = 6 + j * 2
                    predicted_covariance = self.model.propagate_uncertainty(predicted_state, inputs[0, input_id:input_id + 2].numpy(), predicted_covariance)
                    predicted_state = self.model.predict(predicted_state, inputs[0, input_id:input_id + 2].numpy())
                prediction_residual = targets.numpy() - predicted_state
                prediction_residual_2d = np.array([prediction_residual[0,0], prediction_residual[0,1], prediction_residual[0,5]]).reshape(3,1)
                residual_covariance_2d = prediction_residual_2d @ prediction_residual_2d.T
                residual_covariance_2d_vec = vectorize_symmetric_mat(residual_covariance_2d)
                # print(horizon_error)
                measurement_covariance = generate_measurement_covariance(predicted_covariance)
                if not np.all(measurement_covariance == 0):
                    stochastic_prediction_error += residual_covariance_2d_vec.T @ np.linalg.inv(measurement_covariance) @ residual_covariance_2d_vec
                counted_pred_counter += 1
        print('total error stochastic error : ', stochastic_prediction_error)
        print('horizons accounted : ', counted_pred_counter)
        return stochastic_prediction_error

    def train_model(self, init_stoch_params, method, bounds, saved_array_path):
        fun = lambda x: self.compute_stoch_error_all_steps(x)
        training_result = minimize(fun, init_stoch_params, method=method, bounds=bounds)
        np.save(saved_array_path, training_result.x)
        return training_result.x