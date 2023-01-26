import math
# from pypointmatcher import pointmatcher, pointmatchersupport
import glob
import numpy as np
import copy
import pandas as pd
# import wmrde
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
from eval.stochastic_trainer import Stochastic_Trainer

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

train_dataset_path = '/home/dominic/repos/norlab_WMRD/data/marmotte/grand_salon_12_12_a/torch_dataset_all.pkl'
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
robot = 'marmotte'
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

if robot == 'marmotte':
    input_space_dataframe = pd.read_pickle('/home/dominic/repos/norlab_WMRD/data/marmotte/input_space/input_space_data.pkl')
    r = input_space_dataframe['calibrated_radius [m]'].to_numpy()[0]
    baseline = input_space_dataframe['calibrated baseline [m]'].to_numpy()[0]
    alpha = 0.8
    alpha_l = 0.9
    alpha_r = 0.9
    y_icr = 0.5
    y_icr_l = 0.5
    y_icr_r = -0.5
    x_icr = 0.1

# ICR_symmetrical
icr_symmetrical = ICR_symmetrical(r, alpha, x_icr, dt)
args = (icr_symmetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
init_params = [0.88, 0.255] # for icr
bounds = [(0, 1.0), (-2.0, 2.0)]
method = 'Nelder-Mead'

# ICR asymmetrical
# icr_asymmetrical = ICR_asymmetrical(r, alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r, dt)
# args = (icr_asymmetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
# init_params = [alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r] # for icr
# bounds = [(0, 1.5), (0, 1.5), (-5.0, 5.0), (0.001, 5.0), (-5.0, -0.001)]
# method = 'Nelder-Mead'

## Enhanced kinematic
## Husky
# body_inertia = 0.8336
# body_mass = 70
## Marmotte
body_inertia = 0.8336
body_mass = 70
init_params = [0.2, 0.2, 0.0, 0.0]
init_stoch_params = [0.01, 0.001, 0.001, 0.01, 0.001, 0.01]
enhanced_kinematic = Enhanced_kinematic(r, baseline, body_inertia, body_mass, init_params, init_stoch_params, dt)
args = (enhanced_kinematic, wmr_train_dl, timesteps_per_horizon, prediction_weights)
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

trained_params_path = 'training_results/marmotte/enhanced_kinematic/grand_salon_a/train_stochastic_all_horizons.npy'
individual_trained_params_array = 'training_results/marmotte/icr_symmetrical/grand_salon_a/train_individual_horizons.npy'
# velocity_skip_array = np.array([[5.0, -2.0], [5.0, -3.0], [5.0, -4.0]])
# wmr_train_dataset.skip_steps_mask(velocity_skip_array)
stoch_trainer = Stochastic_Trainer(model=enhanced_kinematic, init_stoch_params=init_stoch_params, dataloader=wmr_train_dl,
                              timesteps_per_horizon=timesteps_per_horizon, init_pred_covariance=np.zeros((3,3)))
stoch_trainer.train_model(init_stoch_params=init_stoch_params, method=method, bounds=bounds, saved_array_path=trained_params_path)
# model_trainer.train_model(init_params=init_params, method=method, bounds=bounds, saved_array_path=trained_params_path)
# model_trainer.train_model_all_single_steps(init_params=init_params, method=method, bounds=bounds, saved_array_path=individual_trained_params_array)