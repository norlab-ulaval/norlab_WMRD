import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from data_utils.slip_dataset_parser import SlipDatasetParser

torch_ready_dataset_path = '/home/dominic/repos/norlab_WMRD/data/marmotte/ga_hard_snow_25_01_a/torch_dataset_all.pkl'
export_dataset_path = '/home/dominic/repos/norlab_WMRD/data/marmotte/ga_hard_snow_25_01_a/slip_dataset_all.pkl'
powetrain_model_params_path = '/home/dominic/repos/norlab_WMRD/eval/training_results/marmotte/powertrain/ga_hard_snow_a/'
robot = "marmotte"

slip_dataset_parser = SlipDatasetParser(torch_ready_dataset_path=torch_ready_dataset_path,
                                        export_dataset_path=export_dataset_path,
                                        powertrain_model_params_path=powetrain_model_params_path,
                                        robot=robot)

slip_dataset_parser.append_slip_elements_to_dataset()

