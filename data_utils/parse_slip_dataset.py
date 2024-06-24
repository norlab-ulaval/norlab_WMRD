import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

import sys
path_root = pathlib.Path(__file__).parents[1]
sys.path.append(str(path_root))


from data_utils.slip_dataset_parser import SlipDatasetParser

current_workspace_path = pathlib.Path().cwd()

terrain = 'gravel_1'

export_dataset_path = current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/terrain/'slip_dataset_all.pkl'
torch_ready_dataset_path = current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/terrain/'torch_dataset_all.pkl'
powetrain_model_params_path = str(current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/terrain/'trained_params'/'powertrain')+'/'


# robot = "husky" 
# robot = "marmotte"
# robot = "warthog-track"
robot = "warthog-wheel"

slip_dataset_parser = SlipDatasetParser(torch_ready_dataset_path=torch_ready_dataset_path,
                                        export_dataset_path=export_dataset_path,
                                        powertrain_model_params_path=powetrain_model_params_path,
                                        robot=robot)

slip_dataset_parser.append_slip_elements_to_dataset()

