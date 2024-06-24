import pandas as pd
from script.extractors import *

class data_loader():

    def __init__(self,path_2_acceleration_dataset,verbose=True,rate=0.05,n_step=40) -> None:
        
        self.df_acceleration = pd.read_pickle(path_2_acceleration_dataset)
        self.verbose = verbose
        self.rate = rate
        self.n_step = n_step
    def compute_diamond_graph_numbers(self):
        
        
        # Steady state keeping
        df_steady_state = self.df_acceleration[self.df_acceleration['steady_state_mask'] == 1]
        print(df_steady_state.shape)


        # Remove the first window
        df_last_window = df_steady_state.drop_duplicates(subset=['steady_state_mask','calib_step'],keep='last')
        #print(df_last_window.shape)

        print(df_last_window.shape)
        
        cmd_left = np.mean(column_type_extractor(df_last_window, 'cmd_l',verbose=False),axis=1)
        cmd_right = np.mean(column_type_extractor(df_last_window, 'cmd_r',verbose=False),axis=1)
        self.cmd_vel = np.vstack((cmd_left,cmd_right))
        self.body_vel = compute_body_vel_IDD(self.cmd_vel , robot='warthog-wheel')

        self.icp_vel_x = np.mean(column_type_extractor(df_last_window, 'icp_vel_x',verbose=False),axis=1)
        self.icp_vel_yaw = np.mean(column_type_extractor(df_last_window, 'icp_vel_yaw',verbose=False),axis=1)
        self.body_vel_icp_mean = np.vstack((self.icp_vel_x,self.icp_vel_yaw))

        self.raw_icp_vel_x_mean = np.mean(column_type_extractor(df_last_window, 'raw_icp_vel_x',verbose=False),axis=1)
        self.raw_icp_vel_yaw_mean = np.mean(column_type_extractor(df_last_window, 'raw_icp_vel_yaw',verbose=False),axis=1)


        self.body_vel = compute_body_vel_IDD(self.cmd_vel , robot='warthog-wheel')

        self.odom_speed_l = np.mean(column_type_extractor(df_last_window, 'left_wheel_vel',verbose=False),axis=1)
        self.odom_speed_right = np.mean(column_type_extractor(df_last_window, 'right_wheel_vel',verbose=False),axis=1)
            
    def extract_steady_states_ground_truth_results(self):

        left_wheel_vel = column_type_extractor(self.df_acceleration, 'left_wheel_vel',verbose=self.verbose)
        right_wheel_vel = column_type_extractor(self.df_acceleration, 'right_wheel_vel',verbose=self.verbose)
        icp_vel_x = column_type_extractor(self.df_acceleration, 'icp_vel_x',verbose=self.verbose)
        icp_vel_yaw = column_type_extractor(self.df_acceleration, 'icp_vel_yaw',verbose=self.verbose)
        body_vel_icp = np.vstack((icp_vel_x,icp_vel_yaw))

        column_to_reshape = [left_wheel_vel,right_wheel_vel,
                            icp_vel_x,icp_vel_yaw, body_vel_icp]


        for i in range(len(column_to_reshape)):
            first_index = column_to_reshape[i].shape[0]//3
            total_info = column_to_reshape[i].shape[0]* column_to_reshape[i].shape[1]
            second_size = total_info//first_index
            reshape_size = (first_index, second_size)
            column_to_reshape[i] = column_to_reshape[i].reshape(reshape_size)

        self.left_wheel_vel,self.right_wheel_vel, self.icp_vel_x,self.icp_vel_yaw, self.body_vel_icp = column_to_reshape

        self.time_axis = create_time_axe(self.rate,self.n_step)


    def compute_steady_state_cmd(self):
        cmd_left_all = column_type_extractor(self.df_acceleration, 'cmd_l',verbose=False)
        cmd_right_all = column_type_extractor(self.df_acceleration, 'cmd_r',verbose=False)

        cmd_body_vel_yaw_all = []
        cmd_body_vel_x_all = []

        print(cmd_right_all.shape)
        n_step = cmd_left_all.shape[0] 
        horizon_length = cmd_left_all.shape[1]
        total_info = n_step * horizon_length
        final_shape = ( total_info//(horizon_length *3), horizon_length *3)

        # Compute the body 
        for calib_step in range(n_step):
            for i in range(horizon_length): 
                cmd_vel_wheel_i = np.array([cmd_left_all[calib_step,i],cmd_right_all[calib_step,i]])
            
                body_vel_i = compute_body_vel_IDD(cmd_vel_wheel_i , robot='warthog-wheel')
                cmd_body_vel_yaw_all.append(body_vel_i[1])
                cmd_body_vel_x_all.append(body_vel_i[0])

        self.cmd_body_vel_x_all = np.array(cmd_body_vel_x_all).reshape(final_shape)
        self.cmd_body_vel_yaw_all = np.array(cmd_body_vel_yaw_all).reshape(final_shape)
        self.cmd_left_all_reshape = cmd_left_all.reshape(final_shape)
        self.cmd_right_all_reshape = cmd_right_all.reshape(final_shape)

    def compute_normalized_piece(self):
        self.normalized_left_wheel,self.normalized_cmd_left_all_reshape = normalizer_2d_array(self.left_wheel_vel,self.cmd_left_all_reshape)
        self.normalized_right_wheel_vel,self.normalized_cmd_right_all_reshape = normalizer_2d_array(self.right_wheel_vel,self.cmd_right_all_reshape)
        self.normalized_icp_vel_x, self.normalized_cmd_body_vel_x_all = normalizer_2d_array(self.icp_vel_x, self.cmd_body_vel_x_all)
        self.normalized_icp_vel_yaw, self.normalized_cmd_body_vel_yaw_all = normalizer_2d_array(self.icp_vel_yaw, self.cmd_body_vel_yaw_all)

    def compute_operation_points(self):
        self.operation_point_left_wheel,self.steps_cmd_left_all_reshape = compute_operation_points_and_step(self.left_wheel_vel,self.cmd_left_all_reshape)
        self.operation_point_right_wheel_vel,self.steps_cmd_right_all_reshape = compute_operation_points_and_step(self.right_wheel_vel,self.cmd_right_all_reshape)
        self.operation_point_icp_vel_x, self.steps_cmd_body_vel_x_all = compute_operation_points_and_step(self.icp_vel_x, self.cmd_body_vel_x_all)
        self.operation_point_icp_vel_yaw, self.steps_cmd_body_vel_yaw_all = compute_operation_points_and_step(self.icp_vel_yaw, self.cmd_body_vel_yaw_all)

    def process_the_acceleration_dataframe(self):
        self.compute_diamond_graph_numbers()
        self.extract_steady_states_ground_truth_results()
        self.compute_steady_state_cmd()
        self.compute_normalized_piece()
        self.compute_operation_points()