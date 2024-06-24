import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib


from data_utils.slip_dataset_parser import SlipDatasetParser

current_workspace_path = pathlib.Path().cwd()


export_dataset_path = current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/'gravel_1'/'slip_dataset_all.pkl'
torch_ready_dataset_path = current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/'gravel_1'/'torch_dataset_all.pkl'
powetrain_model_params_path = str(current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/'gravel_1'/'trained_params'/'powertrain')+'/'


# robot = "husky" 
# robot = "marmotte"
# robot = "warthog-track"
robot = "warthog-wheel"

slip_dataset_parser = SlipDatasetParser(torch_ready_dataset_path=torch_ready_dataset_path,
                                        export_dataset_path=export_dataset_path,
                                        powertrain_model_params_path=powetrain_model_params_path,
                                        robot=robot)

slip_dataset_parser.append_slip_elements_to_dataset()

