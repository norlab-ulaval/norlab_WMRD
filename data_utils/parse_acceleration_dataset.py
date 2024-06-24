import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

import sys
path_root = pathlib.Path(__file__).parents[1]
sys.path.append(str(path_root))


from data_utils.acceleration_dataset_parser import AccelerationDatasetParser

current_workspace_path = pathlib.Path().cwd()

terrain = 'gravel_1'
slip_dataset_path = current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/terrain/'slip_dataset_all.pkl'
export_dataset_path = current_workspace_path/'data'/'ral2023_dataset'/'warthog_wheels'/terrain/'acceleration_dataset.pkl'

# robot = "husky"
# robot = "marmotte"
# robot = "warthog-track"
robot = "warthog-wheel"

acceleration_dataset_parser = AccelerationDatasetParser(slip_dataset_path=slip_dataset_path,
                                        export_dataset_path=export_dataset_path,
                                        robot=robot)

acceleration_dataset_parser.append_acceleration_elements_to_dataset()