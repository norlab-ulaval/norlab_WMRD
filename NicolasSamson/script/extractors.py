import numpy as np 
import pandas as pd
from models.kinematic.ideal_diff_drive import Ideal_diff_drive
from scipy.optimize import minimize


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
    print(cmd_2d_array.shape)
    operation_point = (res_2d_array[:,0:5].mean(axis=1)).reshape((res_2d_array.shape[0],1))
    command_abso = (cmd_2d_array[:,-5:].mean(axis=1)).reshape((res_2d_array.shape[0],1)) 
    
    
    
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


class FirstOrderModelWithDelay():

    def __init__(self,initial_gain,initial_time_constant, inital_delay) -> None:
        
        self.X =initial_gain,initial_time_constant,inital_delay 
        
    def foptd(self,t, K=1, tau=1, tau_d=0):
        #tau_d = max(0,tau_d)
        #tau = max(0,tau)
        return np.array([K*(1-np.exp(-(time-tau_d)/tau)) if time >= tau_d else 0 for time in t])

    def err(self,X,t,y):
        K,tau,tau_d = X
        z = self.foptd(t,K,tau,tau_d)
        iae = sum(abs(z-y)**2)
        return iae
    

    

    def train(self,t,u_step,y_centered):
        self.ts = t - t[0]
        self.us = u_step
        ys = y_centered/self.us
        #print(ys.shape)
        #print(self.ts)

        
        self.K,self.tau,self.tau_d = minimize(self.err,self.X,args=(self.ts,ys)).x

        
        return self.K, self.tau, self.tau_d, # gain, time cosntant, delay, step, point operation
    
    def train_all_calib_state(self,time_vec, u_step_array, y_centered_array,operatio_points):

        n_step = u_step_array.shape[0]
        time_constants_computed = np.zeros((n_step,1))
        time_delay_computed = np.zeros((n_step))
        gains_computed = np.zeros((n_step))

        predictions = np.zeros(y_centered_array.shape)

        for calib_step in range(n_step):
            t = time_vec
            #y = y_array[calib_step]
            u_step = u_step_array[calib_step]
            
            y_centered = y_centered_array[calib_step,:]
            operation_point= operatio_points[calib_step]
            gains_computed[calib_step], time_constants_computed[calib_step], time_delay_computed[calib_step] =  self.train(t,u_step,y_centered)

            predictions[calib_step,:] = self.ypred(operation_point)
        return gains_computed, time_constants_computed, time_delay_computed,predictions
    
    def ypred(self,y_operation_pont):


        z = self.foptd(self.ts,self.K,self.tau,self.tau_d)
        ypred = y_operation_pont + z*self.us

        return ypred