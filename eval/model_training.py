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

train_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/masked_datasets/grass_1_right.csv'
training_horizon = 2 # seconds
timestep = 0.05 # seconds
timesteps_per_horizon = int(training_horizon / timestep)

wmr_train_dataset = TorchWMRDataset(train_dataset_path, body_or_wheel_vel='wheel', training_horizon=training_horizon)
wmr_train_dl = DataLoader(wmr_train_dataset)

prediction_weights = np.eye(6)
prediction_weights_2d = np.zeros((6,6))
prediction_weights_2d[0,0] = prediction_weights_2d[1,1] = prediction_weights_2d[5,5] = 1

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
# icr_symmetrical = ICR_symmetrical(r, alpha, y_icr, dt)
# args = (icr_symmetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
# init_params = [0.5, 0.5] # for icr
# bounds = [(0, 1.0), (-5.0, 5.0)]

# ICR assymetrical
icr_assymetrical = ICR_asymmetrical(r, alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r, dt)
args = (icr_assymetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
init_params = [alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r] # for icr
bounds = [(0, 1.0), (0, 1.0), (-5.0, 5.0), (0.0, 5.0), (-5.0, 0.0)]
method = 'Nelder-Mead'

trained_params_path = 'training_results/husky/icr_asymmetrical/doughnut_grass_1_right.npy'

## Enhanced kinematic
# body_inertia = 0.8336
# body_mass = 70
# init_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
# enhanced_kinematic = Enhanced_kinematic(r, baseline, body_inertia, body_mass, init_params, dt)
# args = (enhanced_kinematic, wmr_train_dl, timesteps_per_horizon, prediction_weights)

model_trainer = Model_Trainer(model=icr_assymetrical, init_params=init_params, dataloader=wmr_train_dl,
                              timesteps_per_horizon=timesteps_per_horizon, prediction_weights=prediction_weights_2d)

model_trainer.train_model(init_params=init_params, method=method, bounds=bounds, saved_array_path=trained_params_path)
