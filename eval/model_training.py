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
from eval.model_trainer import Model_Trainer

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

train_dataset_path = '/home/dominic/repos/norlab_WMRD/data/warthog_wheel/doughnut_datasets/depot_2/torch_dataset_all.csv'
# train_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/vel_mask_array_all.npy'
training_horizon = 2 # seconds
timestep = 0.05 # seconds
timesteps_per_horizon = int(training_horizon / timestep)

wmr_train_dataset = TorchWMRDataset(train_dataset_path, body_or_wheel_vel='wheel', training_horizon=training_horizon)

# quadran = 4
# wmr_train_dataset.set_quadran_mask(quadran)



wmr_train_dl = DataLoader(wmr_train_dataset)

prediction_weights = np.eye(6)
prediction_weights_2d = np.zeros((6,6))
prediction_weights_2d[0,0] = prediction_weights_2d[1,1] = prediction_weights_2d[5,5] = 1

#import models
dt = 0.05
# r = 0.5/2
robot = 'warthog-wheel'
if robot == 'husky':
    r = 0.33/2
    baseline = 0.55
    alpha = 0.8
    alpha_l = 0.5
    alpha_r = 0.5
    y_icr = 1.0
    y_icr_l = 1.0
    y_icr_r = -1.0
    x_icr = -1.0
if robot == 'warthog-wheel':
    r = 0.3
    baseline = 1.1652
    alpha = 0.8
    alpha_l = 0.5
    alpha_r = 0.5
    y_icr = 1.0
    y_icr_l = 1.0
    y_icr_r = -1.0
    x_icr = -1.0
if robot == 'warthog-track':
    r = 0.175
    baseline = 1.1652
    alpha = 0.8
    alpha_l = 0.5
    alpha_r = 0.5
    y_icr = 1.0
    y_icr_l = 1.0
    y_icr_r = -1.0
    x_icr = -1.0
alpha_params = np.full((13), 1.0)

## ICR_symmetrical
# icr_symmetrical = ICR_symmetrical(r, alpha, y_icr, dt)
# args = (icr_symmetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
# init_params = [0.5, 0.5] # for icr
# bounds = [(0, 1.0), (-5.0, 5.0)]

# ICR assymetrical
icr_assymetrical = ICR_asymmetrical(r, alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r, dt)
args = (icr_assymetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
init_params = [alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r] # for icr
bounds = [(0, 1.0), (0, 1.0), (-5.0, 5.0), (0.001, 5.0), (-5.0, -0.001)]
method = 'Nelder-Mead'

trained_params_path = 'training_results/warthog_wheel/icr_asymmetrical/depot_2/steady-state/train_full.npy'
# velocity_skip_array = np.array([[5.0, -2.0], [5.0, -3.0], [5.0, -4.0]])
# wmr_train_dataset.skip_steps_mask(velocity_skip_array)
model_trainer = Model_Trainer(model=icr_assymetrical, init_params=init_params, dataloader=wmr_train_dl,
                              timesteps_per_horizon=timesteps_per_horizon, prediction_weights=prediction_weights_2d)
model_trainer.train_model_single_step(init_params, method=method, bounds=bounds, saved_array_path=trained_params_path, step_id=90)
# model_trainer.train_model(init_params=init_params, method=method, bounds=bounds, saved_array_path=trained_params_path)


## Enhanced kinematic
# body_inertia = 0.8336
# body_mass = 70
# init_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
# enhanced_kinematic = Enhanced_kinematic(r, baseline, body_inertia, body_mass, init_params, dt)
# args = (enhanced_kinematic, wmr_train_dl, timesteps_per_horizon, prediction_weights)

# vx_center = 0
# vx_interval = 1.0
# omega_center = 0
# omega_interval = 1.0
# wmr_train_dataset.set_area_mask(vx_center, omega_center, vx_interval, omega_interval)
#
# min_lin_speed = -2.0
# max_lin_speed = 2.0
# lin_speed_step = 0.5
#
# max_ang_speed = 2.5
# n_ang_steps = 12
#
# n_lin_steps = int(max_lin_speed - min_lin_speed / lin_speed_step) + 1
# ang_step = 2 * max_ang_speed / n_ang_steps
#
# vx_interval = 1.0
# omega_interval = 1.0
# for i in range(0, n_lin_steps):
#     vx_center = min_lin_speed + i * lin_speed_step
#     for j in range(0, n_ang_steps + 1):
#         omega_center = -max_ang_speed + j * ang_step
#         wmr_train_dataset.set_area_mask(vx_center, omega_center, vx_interval, omega_interval)
#
#         model_trainer = Model_Trainer(model=icr_assymetrical, init_params=init_params, dataloader=wmr_train_dl,
#                               timesteps_per_horizon=timesteps_per_horizon, prediction_weights=prediction_weights_2d)
#         trained_params_path = 'training_results/husky/icr_asymmetrical/grass/steady-state/areas_1x1/' \
#                               + str(i) + '_' + str(j)
#         model_trainer.train_model(init_params=init_params, method=method, bounds=bounds, saved_array_path=trained_params_path)
