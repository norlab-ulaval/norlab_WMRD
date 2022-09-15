import numpy as np
import pandas as pd
import torch

from eval.torch_dataset import TorchWMRDataset

from scipy.optimize import minimize

from util.util_func import *

class Model_Trainer:
    def __init__(self, model, init_params, dataloader, timesteps_per_horizon, prediction_weights):
        self.model = model
        self.dataloader = dataloader
        self.timesteps_per_horizon = timesteps_per_horizon
        self.prediction_weights = prediction_weights
        self.params = init_params

    def update_dataloader(self, new_dataloader):
        self.dataloader = new_dataloader

    def update_prediction_weights(self, new_prediction_weights):
        self.prediction_weights = new_prediction_weights

    def update_params(self, new_params):
        self.model.adjust_motion_params(new_params)

    def compute_model_error_all_steps(self, init_params):
        self.update_params(init_params)
        # print(init_params)
        prediction_error = 0
        counted_pred_counter = 0
        # self.x_train[idx], self.y_train[idx], self.calib_step[idx], self.mask[idx], self.cmd_vx[idx], self.cmd_omega[idx], \
        #                self.encoder_vx[idx], self.encoder_omega[idx], self.icp_vx[idx], self.icp_vy[idx], self.icp_omega[idx]
        for i, (inputs, targets, step, cmd_vx, cmd_omega, encoder_vx, encoder_omega, icp_vx, icp_vy, icp_omega, steady_state_mask, calib_mask) in enumerate(self.dataloader):
            # print(inputs)
            # print(targets)
            predicted_state = inputs[0, :6].numpy()
            steady_state_mask_bool = steady_state_mask.numpy()
            calib_mask_bool = calib_mask.numpy()
            if steady_state_mask_bool and calib_mask_bool:
            # if True:
                for j in range(0, self.timesteps_per_horizon):
                    input_id = 6 + j * 2
                    predicted_state = self.model.predict(predicted_state, inputs[0, input_id:input_id + 2].numpy())
                horizon_error = disp_err(predicted_state.reshape((6, 1)), targets.numpy().reshape((6, 1)),
                                         self.prediction_weights)
                # print(horizon_error)
                prediction_error += horizon_error
                counted_pred_counter += 1
        # print('total error : ', prediction_error)
        # print('horizons accounted : ', counted_pred_counter)
        return prediction_error

    def train_model(self, init_params, method, bounds, saved_array_path):
        fun = lambda x: self.compute_model_error_all_steps(x)
        training_result = minimize(fun, init_params, method=method, bounds=bounds)
        np.save(saved_array_path, training_result.x)
        return training_result.x

    def train_model_single_step(self, init_params, method, bounds, step_id):
        self.dataloader.dataset.single_step_mask(step_id)
        fun = lambda x: self.compute_model_error_all_steps(x)
        training_result = minimize(fun, init_params, method=method, bounds=bounds)
        return training_result.x

    def train_model_all_single_steps(self, init_params, method, bounds, saved_array_path):
        n_horizons = len(self.dataloader)
        n_params = len(init_params)
        trained_params_array = np.zeros((n_horizons, n_params))
        for i in range(0, n_horizons):
            if self.dataloader.dataset.steady_state_mask[i].numpy() == True:
                print(i, '/', n_horizons)
                print('cmd_vx: ', self.dataloader.dataset.cmd_vx[i].numpy())
                print('cmd_omega: ', self.dataloader.dataset.cmd_omega[i].numpy())
                trained_params_array[i, :] = self.train_model_single_step(init_params, method, bounds, step_id=i)
                print(trained_params_array[i, :])
            else:
                trained_params_array[i, :] = np.full(n_params, None)
        np.save(saved_array_path, trained_params_array)
