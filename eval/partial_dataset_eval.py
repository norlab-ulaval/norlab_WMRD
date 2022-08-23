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
from models.kinematic.enhanced_kinematic import *
from models.kinematic.unicycle import Unicycle
from util.util_func import *

from eval.torch_dataset import TorchWMRDataset
# from eval.model_eval import compute_evaluation

def compute_prediction_error(init_params, model, dataloader, timesteps_per_horizon, prediction_weights):
    model.adjust_motion_params(init_params)
    print(init_params)
    prediction_error = 0
    for i, (inputs, targets, step) in enumerate(dataloader):
        # print(inputs)
        # print(targets)
        predicted_state = inputs[0, :6].numpy()
        for j in range(0, timesteps_per_horizon):
            input_id = 6+j*2
            predicted_state = model.predict(predicted_state, inputs[0, input_id:input_id+2].numpy())
        horizon_error = disp_err(predicted_state.reshape((6,1)), targets.numpy().reshape((6,1)), prediction_weights_2d)
        # print(horizon_error)
        prediction_error += horizon_error
    print('total error : ', prediction_error , '[m]')
    return prediction_error

#import models
dt = 0.05
r = 0.33/2
# r = 0.5/2
baseline = 0.55
body_inertia = 0.8336
body_mass = 70

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

partial_eval_training_params = np.load('training_results/husky/icr_symmetric_per_step/trained_params_per_step.npy')
full_vels_array = np.load('/home/dominic/repos/norlab_WMRD/data/husky/calibration_vels_array.npy')
print(partial_eval_training_params[:, :, 0])
print(full_vels_array.shape)

icr_symmetrical = ICR_symmetrical(r, partial_eval_training_params[0,0,0], partial_eval_training_params[0,0,1], dt)
unicycle = Unicycle(dt)

prediction_weights = np.eye(6)
prediction_weights_2d = np.zeros((6,6))
prediction_weights_2d[0,0] = prediction_weights_2d[1,1] = prediction_weights_2d[5,5] = 1

prediction_errors = np.zeros((full_vels_array.shape[0], full_vels_array.shape[1]))

for i in range(0, full_vels_array.shape[0]):
    for j in range(0, full_vels_array.shape[1]):
        prediction_errors[i,j] = compute_prediction_error(partial_eval_training_params[i,j,:], icr_symmetrical, wmr_eval_dl,
                                                      timesteps_per_horizon, prediction_weights_2d)

plt.figure(figsize=(10,10))
ax = plt.gca()
fig = plt.gcf()

alpha_plot = 1.0

cmd = ax.scatter(full_vels_array[:, :, 1].flatten(), full_vels_array[:, :, 0].flatten(),
                  c = prediction_errors.flatten(),
                  cmap = 'hot',
                  alpha = alpha_plot,
                  lw=0,
                  s=50,
                  label='Body vel commands',
                 rasterized=True)

ax.set_xlabel('Ignored commanded angular velocity [rad]')
ax.set_ylabel('Ignored commanded linear velocity [m]')

fig.colorbar(cmd, label='Residual prediction error')

plt.show()