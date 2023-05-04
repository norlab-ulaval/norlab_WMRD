import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from data_utils.slip_dataset_parser import SlipDatasetParser

torch_ready_dataset_path = '/home/dominic/repos/norlab_WMRD/data/ral2023_dataset/warthog_tracks/grand-axe_crusted-snow/torch_dataset_all.pkl'
export_dataset_path = '/home/dominic/repos/norlab_WMRD/data/ral2023_dataset/warthog_tracks/grand-axe_crusted-snow/slip_dataset_all.pkl'
powetrain_model_params_path = '/home/dominic/repos/norlab_WMRD/data/ral2023_dataset/warthog_tracks/grand-axe_crusted-snow/trained_params/powertrain/'
# robot = "husky"
# robot = "marmotte"
robot = "warthog-track"

slip_dataset_parser = SlipDatasetParser(torch_ready_dataset_path=torch_ready_dataset_path,
                                        export_dataset_path=export_dataset_path,
                                        powertrain_model_params_path=powetrain_model_params_path,
                                        robot=robot)

slip_dataset_parser.append_slip_elements_to_dataset()

