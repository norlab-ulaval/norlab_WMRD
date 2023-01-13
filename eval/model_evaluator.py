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
        self.n_evaluated_horizons = len(dataloader)

        self.body_to_world_tf = np.eye(6)
        self.body_to_world_rotmat = np.eye(2)

        self.prediction_weights_6dof = np.eye(6)
        self.prediction_weights_3dof = np.zeros((6,6))
        self.prediction_weights_3dof[0,0] = 1
        self.prediction_weights_3dof[1,1] = 1
        self.prediction_weights_3dof[5,5] = 1
        self.prediction_weights_3dof_trans = np.zeros((6,6))
        self.prediction_weights_3dof_trans[0,0] = 1
        self.prediction_weights_3dof_trans[1,1] = 1
        self.prediction_weights_3dof_ang = np.zeros((6,6))
        self.prediction_weights_3dof_ang[5,5] = 1

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
        # body_commands_array = np.full((n_total_horizons, 3), None)
        # body_encoder_array = np.full((n_total_horizons, 3), None)
        icp_vels_array = np.full((n_total_horizons, 3), None)
        model_body_vels = np.zeros(3)
        model_body_vels_array = np.full((n_total_horizons, 3), None)

        for i, (inputs, targets, step, icp_vx, icp_vy, icp_omega, steady_state_mask, calib_mask) in enumerate(self.dataloader):
            # print(inputs)
            # print(targets)
            prev_predicted_state = inputs[0, :6].numpy()
            predicted_state = inputs[0, :6].numpy()
            euler_pose_to_transform(predicted_state[3:], predicted_state[:3], self.body_to_world_tf)
            yaw_to_rotmat2d(self.body_to_world_rotmat, predicted_state[5])
            steady_state_mask_bool = steady_state_mask.numpy()
            calib_mask_bool = calib_mask.numpy()
            # if steady_state_mask_bool:
            if True:
                for j in range(0, self.timesteps_per_horizon):
                    input_id = 6 + j * 2
                    predicted_state = self.model.predict(predicted_state, inputs[0, input_id:input_id + 2].numpy())
                    model_body_vels[:2] = model_body_vels[:2] + (self.body_to_world_rotmat.T @ \
                                          (predicted_state[:2] - prev_predicted_state[:2])) / self.dt
                    model_body_vels[2] = model_body_vels[2] + ((predicted_state[5] - prev_predicted_state[5]) / self.dt)
                    euler_pose_to_transform(predicted_state[3:], predicted_state[:3], self.body_to_world_tf)
                    yaw_to_rotmat2d(self.body_to_world_rotmat, predicted_state[5])
                    prev_predicted_state = predicted_state
                prediction_error_array[i, :] = targets[:6].numpy() - predicted_state
                prediction_error_array[i, 3] = wrap2pi(prediction_error_array[i, 3])
                prediction_error_array[i, 4] = wrap2pi(prediction_error_array[i, 4])
                prediction_error_array[i, 5] = wrap2pi(prediction_error_array[i, 5])
                # body_commands_array[i, 0] = cmd_vx.numpy()[0]
                # body_commands_array[i, 1] = 0.0
                # body_commands_array[i, 2] = cmd_omega.numpy()[0]
                # body_encoder_array[i, 0] = encoder_vx.numpy()[0]
                # body_encoder_array[i, 1] = 0.0
                # body_encoder_array[i, 2] = encoder_omega.numpy()[0]
                icp_vels_array[i,0] = icp_vx.numpy()[0]
                icp_vels_array[i,1] = icp_vy.numpy()[0]
                icp_vels_array[i,2] = icp_omega.numpy()[0]
                model_body_vels = model_body_vels / self.timesteps_per_horizon
                model_body_vels_array[i, :] = model_body_vels

                counted_pred_counter += 1

        prediction_error_array = prediction_error_array[prediction_error_array[:, 0] != None]
        # body_commands_array = body_commands_array[body_commands_array[:, 0] != None]
        # body_encoder_array = body_encoder_array[body_encoder_array[:, 0] != None]
        icp_vels_array = icp_vels_array[icp_vels_array[:, 0] != None]
        model_body_vels_array = model_body_vels_array[model_body_vels_array[:, 0] != None]

        return prediction_error_array, icp_vels_array, model_body_vels_array

    def compute_disp_error(self, x_err, prediction_weights):
        return x_err.T @ prediction_weights @ x_err

    def compute_then_export_prediction_error_metrics(self, params, export_path):
        prediction_error_array, icp_vels_array, model_body_vels_array = self.compute_model_evaluation_metrics(params)

        self.n_evaluated_horizons = prediction_error_array.shape[0]

        prediction_error_6dof_array = np.full(self.n_evaluated_horizons, None)
        prediction_error_3dof_array = np.full(self.n_evaluated_horizons, None)
        prediction_error_3dof_trans_array = np.full(self.n_evaluated_horizons, None)
        prediction_error_3dof_ang_array = np.full(self.n_evaluated_horizons, None)

        for i in range(0, self.n_evaluated_horizons):
            prediction_error_6dof_array[i] = self.compute_disp_error(prediction_error_array[i, :], self.prediction_weights_6dof)
            prediction_error_3dof_array[i] = self.compute_disp_error(prediction_error_array[i, :], self.prediction_weights_3dof)
            prediction_error_3dof_trans_array[i] = np.linalg.norm(prediction_error_array[i, :2])
            prediction_error_3dof_ang_array[i] = np.abs(prediction_error_array[i, 5])

        full_errors_metric_array = np.concatenate((prediction_error_array.reshape(self.n_evaluated_horizons,6),
                                                   icp_vels_array.reshape(self.n_evaluated_horizons, 3),
                                                   model_body_vels_array.reshape(self.n_evaluated_horizons, 3),
                                                   prediction_error_6dof_array.reshape(self.n_evaluated_horizons, 1),
                                                   prediction_error_3dof_array.reshape(self.n_evaluated_horizons, 1),
                                                   prediction_error_3dof_trans_array.reshape(self.n_evaluated_horizons, 1),
                                                   prediction_error_3dof_ang_array.reshape(self.n_evaluated_horizons, 1)),
                                                  axis=1)

        cols = ['prediction_error_x', 'prediction_error_y', 'prediction_error_z',
                'prediction_error_roll', 'prediction_error_pitch', 'prediction_error_yaw',
                'body_icp_vx', 'body_icp_vy', 'body_icp_omega',
                'body_model_vx', 'body_model_vy', 'body_model_omega',
                'prediction_error_6dof', 'prediction_error_3dof',
                'prediction_error_3dof_trans', 'prediction_error_3dof_ang']

        full_errors_metric_dataframe = pd.DataFrame(full_errors_metric_array, columns=cols)
        full_errors_metric_dataframe.to_pickle(export_path)