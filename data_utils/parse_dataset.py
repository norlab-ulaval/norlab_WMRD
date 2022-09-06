import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils.dataset_parser import DatasetParser

raw_dataset_path = '/media/dominic/462D-1EE8/rosbags/doughnut_calibs/husky/day2/doughnut/extracted_data/doughnut_grass2.csv'
export_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/doughnut_datasets/grass_2/torch_dataset_all.csv'

robot = 'husky'

if robot == 'husky':
    steady_state_step_len = 160
    wheel_radius = 0.33/2
    baseline = 0.55
    training_horizon = 2
    rate = 0.05

if robot == 'warthog-wheel':
    steady_state_step_len = 160
    wheel_radius = 0.3
    baseline = 1.1652
    training_horizon = 2
    rate = 0.05

if robot == 'warthog-track':
    steady_state_step_len = 160
    wheel_radius = 0.175
    baseline = 1.1652
    training_horizon = 2
    rate = 0.05

dataset_parser = DatasetParser(raw_dataset_path, export_dataset_path, training_horizon, steady_state_step_len, wheel_radius, baseline, rate)

dataset_parser.process_data(2.5, -2.5, 2.5, -2.5)