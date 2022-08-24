import numpy as np
import pandas as pd
import torch

from eval.torch_dataset import TorchWMRDataset

from util.util_func import *
from util.transform_algebra import *

class Model_Evaluator:
    def __init__(self, model, params, dataset, dataloader, timesteps_per_horizon, prediction_weights):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.timesteps_per_horizon = timesteps_per_horizon
        self.dt = self.model.dt
        self.horizon_length = self.timesteps_per_horizon * self.dt
        self.prediction_weights = prediction_weights
        self.params = params

        self.body_to_world_tf = np.eye(6)
        self.body_to_world_rotmat = np.eye(2)

    def update_dataloader(self, new_dataloader):
        self.dataloader = new_dataloader

    def update_prediction_weights(self, new_prediction_weights):
        self.prediction_weights = new_prediction_weights

    def update_params(self, new_params):
        self.model.adjust_motion_params(new_params)

    def compute_model_evaluation_metrics(self, params):
        self.update_params(params)
        counted_pred_counter = 0
        n_total_horizons = self.dataset.__len__()
        valid_training_id_counter = 0

        prediction_error_array = np.full((n_total_horizons, 6), None)
        body_commands_array = np.full((n_total_horizons, 3), None)
        body_encoder_array = np.full((n_total_horizons, 3), None)
        icp_vels_array = np.full((n_total_horizons, 3), None)
        model_body_vels = np.zeros(3)
        model_body_vels_array = np.full((n_total_horizons, 3), None)

        for i, (inputs, targets, step, mask, cmd_vx, cmd_omega, encoder_vx, encoder_omega, icp_vx, icp_vy, icp_omega) in enumerate(self.dataloader):
            # print(inputs)
            # print(targets)
            prev_predicted_state = inputs[0, :6].numpy()
            predicted_state = inputs[0, :6].numpy()
            euler_pose_to_transform(predicted_state[3:], predicted_state[:3], self.body_to_world_tf)
            yaw_to_rotmat2d(self.body_to_world_rotmat, predicted_state[5])
            mask_bool = mask.numpy()
            if mask_bool:
                for j in range(0, self.timesteps_per_horizon):
                    input_id = 6 + j * 2
                    predicted_state = self.model.predict(predicted_state, inputs[0, input_id:input_id + 2].numpy())
                    model_body_vels[:2] = model_body_vels[:2] + self.body_to_world_rotmat @ \
                                          (predicted_state[:2] - prev_predicted_state[:2])
                    model_body_vels[2] = model_body_vels[2] + (predicted_state[5] - prev_predicted_state[5])
                    euler_pose_to_transform(predicted_state[3:], predicted_state[:3], self.body_to_world_tf)
                    yaw_to_rotmat2d(self.body_to_world_rotmat, predicted_state[5])
                    prev_predicted_state = predicted_state
                prediction_error_array[i, :] = targets[:6].numpy() - predicted_state
                body_commands_array[i, 0] = cmd_vx.numpy()
                body_commands_array[i, 2] = cmd_omega.numpy()
                body_encoder_array[i, 0] = encoder_vx.numpy()
                body_encoder_array[i, 2] = encoder_omega.numpy()
                icp_vels_array[i,0] = icp_vx.numpy()
                icp_vels_array[i,1] = icp_vy.numpy()
                icp_vels_array[i,2] = icp_omega.numpy()
                model_body_vels = model_body_vels / self.timesteps_per_horizon
                model_body_vels_array[i, :] = model_body_vels

                counted_pred_counter += 1

        return prediction_error_array, body_commands_array, body_encoder_array, icp_vels_array, model_body_vels_array