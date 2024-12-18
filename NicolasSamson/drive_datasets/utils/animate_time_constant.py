import numpy as np
import pandas as pd
import os
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
from NicolasSamson.drive_datasets.utils.extractors import * 
from util.transform_algebra import *
from util.util_func import *
from data_utils.dataset_parser import *


import matplotlib.animation as animation
from matplotlib.backend_bases import KeyEvent



# Update the line data
def update(anim_i,time_axis,line,line2,line3,predictions,gt_of_interest_reshpae,cmd_of_interest_reshape, ax,names):
    line.set_data(time_axis, predictions[anim_i,:])
    line2.set_data(time_axis, gt_of_interest_reshpae[anim_i,:])
    line3.set_data(time_axis, cmd_of_interest_reshape[anim_i,:])
    #ax.set_title(f"time constant {time_constants_computed[anim_i]} \n time_delay {time_delay_computed} \n gains {gains_computed} ")
    ax.set_title(f"{names[0]} Step={anim_i}")
    return line,line2,line3

# Function to handle key presses
def produce_video(predictions,time_axis,cmd_of_interest_reshape,gt_of_interest_reshpae ,names=["cmd","model","measurement"],video_saving_path=""): 

    # Initialize data
    data = {'x': [], 'y': []}

    # Create a figure and axis
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2,label ="Model")
    line2, = ax.plot([],[],label = "GT")
    line3, = ax.plot([],[],label="Cmd")
    # Set up the plot limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-20, 20)
    ax.legend()

    anim_i = 0

    ani = animation.FuncAnimation(fig, update,fargs=[time_axis,line,line2,line3,predictions,gt_of_interest_reshpae,cmd_of_interest_reshape, ax,names], frames=predictions.shape[0], interval=1000, blit=True)

    # Save the animation as a video
    final_path_2_save_video = f'{names[0]}_step_by_step_visualisation.mp4'

    if video_saving_path!="":
        path_2_save_video = os.path.join((video_saving_path,final_path_2_save_video))

    
    ani.save(path_2_save_video, writer='ffmpeg')

