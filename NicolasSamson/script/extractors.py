import numpy as np 
import pandas as pd
from models.kinematic.ideal_diff_drive import Ideal_diff_drive

def print_column_unique_column(df):
    df_columns = list(df.columns)
    possible_number = ["1","2","3","4","5","6","7","8","9","0"]
    for i,column in enumerate(df_columns):

        if column[-1] in possible_number:

            if column[-2] in possible_number:
                df_columns[i] = column[:-3]
            else:
                df_columns[i] = column[:-2]

    df_columns_name = pd.Series(df_columns)
    print(df_columns_name.unique())

def column_type_extractor(df, common_type,
                        transient_state=True,steady_state=True, verbose=False):
    """ Extract the np.matrix that represent the 40 columns of all steps of the 
    specific type. 
    column_type_extractor
    For example, icp_velx_40 is of the type icp_velx 
    
    """
    columns_mask = df.columns.str.startswith(common_type)
    column_to_take = df.columns[columns_mask]

    local_df = df.copy()
    if transient_state==False and steady_state==True:
        mask = local_df.steady_state_mask == 1
        local_df = local_df.loc[mask]
    elif steady_state == False and transient_state == True:
        mask = local_df.steady_state_mask == 0
        local_df = local_df.loc[mask]
    elif steady_state == False and transient_state == False:
        raise ValueError("Both steady state and transient can not be at false")
    
    np_results = local_df[column_to_take].to_numpy().astype('float')

    if verbose == True:
        message = "_"*8+f"{common_type}"+"_"*8
        print(message)
        print(f"The column type: {common_type}")
        print(f"The resulting dataframe_shape: {np_results.shape}")
        print(f"Number of calibrating steps:{np_results.shape[0]}")
        print(f"Number of measurement by step: {np_results.shape[1]}")
        print(f"Maximum {np.max(np_results)}")
        print(f"Minimum {np.min(np_results)}")
        print("_"*len(message))

    return np_results

def create_time_axe(rate,n):
    
    return np.array(range(0,n)) * rate

def compute_body_vel_IDD( u, robot='warthog-wheel'):
    if robot == 'warthog-wheel':
        wheel_radius = 0.3
        baseline = 1.1652
        
        rate = 0.05

        model = Ideal_diff_drive(wheel_radius,baseline,rate)
    
    
    body_vel = model.compute_body_vel(u)

    return body_vel

def compute_operation_points_and_step(res_2d_array,cmd_2d_array):
    noramlizer_index = 1
    operation_point = (res_2d_array[:,0:4].mean(axis=1)).reshape((res_2d_array.shape[0],1))
    command_abso = cmd_2d_array[:,noramlizer_index].reshape((res_2d_array.shape[0],1)) 
    
    steps = command_abso - operation_point

    return operation_point,steps


def normalizer_2d_array(res_2d_array,cmd_2d_array):
    """To normalize the step answer by the first column of the command vector
    Args:
        res_2d_array (_type_): the nb_calibration_stepxhorizon_length(40 or 120) AKA 2 or 6second
        cmd_2d_array (_type_): _description_
    Returns:
        _type_: _description_
    """
    operation_point,steps = compute_operation_points_and_step(res_2d_array,cmd_2d_array)

    normalizer_coefficient = 1/steps

    centered_2d_array = res_2d_array - operation_point
    normalize_2d_array = centered_2d_array * normalizer_coefficient
    
    normalized_cmd_2d_array = np.ones(normalize_2d_array.shape)
    return normalize_2d_array,normalized_cmd_2d_array