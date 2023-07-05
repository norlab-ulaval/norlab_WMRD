

# from pypointmatcher import pointmatcher, pointmatchersupport
import pandas as pd
# import wmrde
import torch
from torch.utils.data import DataLoader
# from torchmin import minimize

from models.kinematic.ideal_diff_drive import Ideal_diff_drive
from models.kinematic.ICR_based import *
from models.kinematic.enhanced_kinematic import *
from util.util_func import *

from eval.torch_dataset import TorchWMRDataset
from eval.model_evaluator import Model_Evaluator

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

train_dataset_path = '/home/dominic/repos/norlab_WMRD/data/marmotte/ga_hard_snow_25_01_b/torch_dataset_all.pkl'
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
    x_icr = 0.0

## ideal_diff-drive
ideal_diff_drive = Ideal_diff_drive(r, baseline, dt)

# ICR_symmetrical
icr_symmetrical = ICR_symmetrical(r, alpha, y_icr, dt)
args = (icr_symmetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
init_params = [1.0, 0.5] # for icr
bounds = [(0, 1.0), (-5.0, 5.0)]

# ICR assymetrical
icr_asymmetrical = ICR_asymmetrical(r, alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r, dt)
args = (icr_asymmetrical, wmr_train_dl, timesteps_per_horizon, prediction_weights)
init_params = [alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r] # for icr
bounds = [(0, 1.5), (0, 1.5), (-5.0, 5.0), (0.0, 5.0), (-5.0, 0.0)]
method = 'Nelder-Mead'

trained_params_path = 'training_results/marmotte/icr_asymmetrical/ga_hard_snow_a/train_full_all_horizons.npy'
trained_params = np.load(trained_params_path)

## Enhanced kinematic
body_inertia = 0.8336
body_mass = 70
init_stoch_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
init_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

enhanced_kinematic = Enhanced_kinematic(r, baseline, body_inertia, body_mass, init_params, init_stoch_params, dt)
args = (enhanced_kinematic, wmr_train_dl, timesteps_per_horizon, prediction_weights)

model_evaluator = Model_Evaluator(model=icr_asymmetrical, params=trained_params, dataset=wmr_train_dataset, dataloader=wmr_train_dl,
                                  timesteps_per_horizon=timesteps_per_horizon, prediction_weights=prediction_weights_2d)

# prediction_error_array, body_commands_array, body_encoder_array, icp_vels_array, model_body_vels_array = \
#     model_evaluator.compute_model_evaluation_metrics(trained_params)

export_path = '../data/marmotte/eval_results/ga_hard_snow_b/icr_asymmetrical_full_eval_metrics.pkl'

model_evaluator.compute_then_export_prediction_error_metrics(trained_params, export_path)
