import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)
plot_fs = 12

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('axes', labelsize=10)
mpl.rcParams['lines.dashed_pattern'] = [2, 2]
mpl.rcParams['lines.linewidth'] = 1.0

import pathlib
cur_dir = pathlib.Path().cwd()
parent_dir = cur_dir.parent

import sys
print(sys.version)
sys.path.append('../')

from models.powertrain.bounded_powertrain import Bounded_powertrain
from models.kinematic.ideal_diff_drive import Ideal_diff_drive
from models.learning.blr_slip import SlipBayesianLinearRegression, FullBodySlipBayesianLinearRegression
from models.learning.blr_slip_acceleration import SlipAccelerationBayesianLinearRegression, FullBodySlipAccelerationBayesianLinearRegression
from models.kinematic.ICR_based import *
from models.kinematic.Perturbed_unicycle import *
from models.kinematic.enhanced_kinematic import *
from NicolasSamson.script.extractors import * 
from util.transform_algebra import *
from util.util_func import *
from data_utils.dataset_parser import *




path_2_dataset = '../data/ral2023_dataset/warthog_wheels/ice/acceleration_dataset.pkl'


df_acceleration = pd.read_pickle(path_2_dataset)

print_column_unique_column(df_acceleration)



icep_vel_x = column_type_extractor(df_acceleration, 'icp_vel_x',verbose=False)
icep_vel_yaw = column_type_extractor(df_acceleration, 'icp_vel_yaw',verbose=False)
time = create_time_axe(0.05,icep_vel_x.shape[1])


fig = plt.figure(figsize=plt.figaspect(0.5))
axs = fig.add_subplot(1, 1, 1, projection='3d')

for i in range(icep_vel_x.shape[1]):
    axs.scatter(icep_vel_x[i,:],icep_vel_yaw[i,:],time)

axs.set_ylabel("Yaw speed [rad/s]")
axs.set_xlabel("X speed [m/s]")

plt.show()