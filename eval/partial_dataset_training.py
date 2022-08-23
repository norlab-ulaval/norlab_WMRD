import math
from pypointmatcher import pointmatcher, pointmatchersupport
import glob
import numpy as np
import copy
import pandas as pd
import wmrde
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
# from torchmin import minimize

from models.kinematic.ICR_based import *
from models.kinematic.Perturbed_unicycle import *
from models.kinematic.enhanced_kinematic import *
from util.util_func import *

from eval.torch_dataset import TorchWMRDataset

from scipy.optimize import minimize

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

train_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_1_torch_wheel_encoder.csv'
training_horizon = 2 # seconds
timestep = 0.05 # seconds
timesteps_per_horizon = int(training_horizon / timestep)

wmr_train_dataset = TorchWMRDataset(train_dataset_path, body_or_wheel_vel='wheel', training_horizon=training_horizon)
wmr_train_dl = DataLoader(wmr_train_dataset)

body_cmd_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_body_cmd.csv'
wmr_body_cmd_dataset = TorchWMRDataset(body_cmd_dataset_path, body_or_wheel_vel='body', training_horizon=training_horizon)


prediction_weights = np.eye(6)
prediction_weights_2d = np.zeros((6,6))
prediction_weights_2d[0,0] = prediction_weights_2d[1,1] = prediction_weights_2d[5,5] = 1
def compute_prediction_error(init_params, model, dataloader, timesteps_per_horizon, prediction_weights, masked_vels):
    model.adjust_motion_params(init_params)
    # print(init_params)
    prediction_error = 0
    for i, (inputs, targets, step) in enumerate(dataloader):
        # print(inputs)
        # print(targets)
        body_vel_command, state, cmd_step = wmr_body_cmd_dataset.__getitem__(i)
        cmd_vx = body_vel_command[6].numpy()
        cmd_omega = body_vel_command[7].numpy()
        if cmd_vx != masked_vels[0] or cmd_omega != masked_vels[1]:
            predicted_state = inputs[0, :6].numpy()
            for j in range(0, timesteps_per_horizon):
                input_id = 6+j*2
                predicted_state = model.predict(predicted_state, inputs[0, input_id:input_id+2].numpy())
            horizon_error = disp_err(predicted_state.reshape((6,1)), targets.numpy().reshape((6,1)), prediction_weights_2d)
            # print(horizon_error)
            prediction_error += horizon_error
    # print('total error : ', prediction_error , '[m]')
    return prediction_error

#import models
dt = 0.05
r = 0.33/2
# r = 0.5/2
baseline = 0.55
alpha = 0.8
alpha_l = 0.5
alpha_r = 0.5
y_icr = 1.0
y_icr_l = 1.0
y_icr_r = -1.0
x_icr = -1.0
alpha_params = np.full((13), 1.0)

## ICR_symmetrical
icr_symmetrical = ICR_symmetrical(r, alpha, y_icr, dt)

init_params = [0.5, 0.5] # for icr
bounds = [(0, 1.0), (-5.0, 5.0)]

# import calib velocities

full_vels_array = np.load('/home/dominic/repos/norlab_WMRD/data/husky/calibration_vels_array.npy')
trained_params_array = np.zeros((full_vels_array.shape[0], full_vels_array.shape[1], 2))

total_vels = full_vels_array.shape[0] * full_vels_array.shape[1]
current_vel_step = 0

# for i in range(0, full_vels_array.shape[0]):
#     for j in range(0, full_vels_array.shape[1]):
#         current_vel_step += 1
#         print(current_vel_step, ' / ', total_vels)
#         masked_vels = full_vels_array[i,j,:]
#         print(masked_vels)
#         args = (icr_symmetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights, masked_vels)
#         trained_params = minimize(compute_prediction_error, init_params, args=args, method='Nelder-Mead', bounds=bounds)
#         trained_params_array[i,j,:] = trained_params.x

# vel_mask_array = np.load('/home/dominic/repos/norlab_WMRD/data/husky/vel_mask_array_all.npy', vel_mask_array)
# masked_vels = full_vels_array[vel_mask_array]
#
# for i in range(masked_vels)
#
# np.save('training_results/husky/icr_symmetric_per_step/trained_params_per_step.npy', trained_params_array)
# print(masked_vels)