import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from model_training import compute_prediction_error

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
# from torchmin import minimize

from models.kinematic.ICR_based import *
from models.kinematic.Perturbed_unicycle import *
from models.kinematic.unicycle import Unicycle
from util.util_func import *

from eval.torch_dataset import TorchWMRDataset

#import models
dt = 0.05
r = 0.33/2
# r = 0.5/2
baseline = 0.55

eval_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_wheel_encoder.csv'
training_horizon = 2 # seconds
timestep = 0.05 # seconds
timesteps_per_horizon = int(training_horizon / timestep)

body_cmd_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_body_cmd.csv'
body_encoder_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_body_encoder.csv'

wmr_eval_dataset = TorchWMRDataset(eval_dataset_path, body_or_wheel_vel='wheel', training_horizon=training_horizon)
wmr_eval_dl = DataLoader(wmr_eval_dataset)

wmr_body_cmd_dataset = TorchWMRDataset(body_cmd_dataset_path, body_or_wheel_vel='body', training_horizon=training_horizon)
wmr_body_encoder_dataset = TorchWMRDataset(body_encoder_dataset_path, body_or_wheel_vel='body', training_horizon=training_horizon)

trained_params = np.load('training_results/husky/doughnut_grass_1.npy')

icr_symmetrical = ICR_symmetrical(r, trained_params[0], trained_params[1], dt)
icr_symmetrical.adjust_motion_params(trained_params)
unicycle = Unicycle(dt)

prediction_weights = np.eye(6)
prediction_weights_2d = np.zeros((6,6))
prediction_weights_2d[0,0] = prediction_weights_2d[1,1] = prediction_weights_2d[5,5] = 1

calib_step_array = np.zeros((wmr_eval_dataset.__len__(), 1))
prediction_errors_array = np.zeros((wmr_eval_dataset.__len__(), 1))

def compute_evaluation(model, unicycle_model, eval_dataset, body_cmd_dataset, body_encoder_dataset, timesteps_per_horizon, prediction_weights):
    n_points = eval_dataset.__len__()
    prediction_errors_array = np.zeros((n_points, 6))
    model_disp_array = np.zeros((n_points, 6))
    body_cmd_disp_array = np.zeros((n_points, 6))
    body_encoder_disp_array = np.zeros((n_points, 6))
    icp_disp_array = np.zeros((n_points, 6))

    for i in range(0, n_points):
        input, target, step = eval_dataset.__getitem__(i)
        body_cmd_input, body_cmd_target, body_cmd_stp = body_cmd_dataset.__getitem__(i)
        body_encoder_input, body_encoder_target, body_encoder_stp = body_encoder_dataset.__getitem__(i)
        input = input.numpy()
        target = target.numpy()
        step = step.numpy()
        body_cmd_input = body_cmd_input.numpy()
        body_encoder_input = body_encoder_input.numpy()
        predicted_state_model = input[:6]
        predicted_state_body_cmd = input[:6]
        predicted_state_body_encoder = input[:6]
        init_state = input[:6]
        for j in range(0, timesteps_per_horizon):
            input_id = 6 + j * 2
            predicted_state_model = model.predict(predicted_state_model, input[input_id:input_id + 2])
            predicted_state_body_cmd = unicycle_model.predict(predicted_state_body_cmd, body_cmd_input[input_id:input_id + 2])
            predicted_state_body_encoder = unicycle_model.predict(predicted_state_body_encoder, body_encoder_input[input_id:input_id + 2])

        prediction_errors_array[i, :] = target - predicted_state_model
        prediction_errors_array[i, 5] = wrap2pi(prediction_errors_array[i, 5])
        body_cmd_disp_array[i, :] = predicted_state_body_cmd - init_state
        body_cmd_disp_array[i, 5] = wrap2pi(body_cmd_disp_array[i, 5])
        body_encoder_disp_array[i, :] = predicted_state_body_encoder - init_state
        body_encoder_disp_array[i, 5] = wrap2pi(body_encoder_disp_array[i, 5])
        icp_disp_array[i, :] = target - init_state
        icp_disp_array[i, 5] = wrap2pi(icp_disp_array[i, 5])
        model_disp_array[i, :] = predicted_state_model - init_state
        model_disp_array[i, 5] = wrap2pi(model_disp_array[i, 5])

        # TODO: Figure out wrap2pi for angular displacement (need to iterate on it) / transform all displacements in body frame...

    return prediction_errors_array, model_disp_array, body_cmd_disp_array, body_encoder_disp_array, icp_disp_array


prediction_errors_array, model_disp_array, body_cmd_disp_array, \
body_encoder_disp_array, icp_disp_array = compute_evaluation(icr_symmetrical, unicycle, wmr_eval_dataset,
                          wmr_body_cmd_dataset, wmr_body_encoder_dataset, timesteps_per_horizon, prediction_weights)

plt.plot(body_cmd_disp_array[:, 5], label='unicycle')
plt.plot(body_encoder_disp_array[:, 5], label='encoder_DD')
plt.plot(model_disp_array[:, 5], label='model')
plt.plot(icp_disp_array[:, 5], label = 'icp')
# plt.plot(prediction_errors_array[:, 5])
plt.legend()
plt.show()