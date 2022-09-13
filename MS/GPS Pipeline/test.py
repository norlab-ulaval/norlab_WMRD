import pandas as pd
import utm
import matplotlib.pyplot as plt
import numpy as np

def convert_list_of_lattitudes_and_longitudes_to_utm_coordinates(lattitudes, longitudes):
    utm_lattitudes = []
    utm_longitudes = []
    for i in range(len(lattitudes)):
        utm_lattitudes.append(utm.from_latlon(lattitudes[i], longitudes[i])[0])
        utm_longitudes.append(utm.from_latlon(lattitudes[i], longitudes[i])[1])
    return utm_lattitudes, utm_longitudes

def plot_utm_coordinates(utm_lattitudes, utm_longitudes):
    plt.plot(utm_lattitudes, utm_longitudes, '.')
    plt.show()

def remove_min_value_from_all_elements_in_list(list_of_values):
    min_value = min(list_of_values)
    print(min_value)
    for i in range(len(list_of_values)):
        list_of_values[i] = list_of_values[i] - min_value
    return list_of_values

def import_gps_data_from_pos_file(csv_file_path):
    gps_data = pd.read_table(csv_file_path, sep="\s+", parse_dates={"Timestamp": [0, 1]}, error_bad_lines=False, skiprows=25)
    return gps_data

def get_utm_lattitudes_and_longitudes_from_gps_data(gps_data):
    
    lattitudes = gps_data.values[:,1].astype(float)
    longitudes = gps_data.values[:,2].astype(float)
    
    utm_lattitudes, utm_longitudes = convert_list_of_lattitudes_and_longitudes_to_utm_coordinates(lattitudes, longitudes)
    
    #utm_lattitudes = remove_min_value_from_all_elements_in_list(utm_lattitudes)
    #utm_longitudes = remove_min_value_from_all_elements_in_list(utm_longitudes)
    
    return utm_lattitudes, utm_longitudes

def resample_utm_lattitudes_and_longitudes(utm_lattitudes, utm_longitudes, target_sampling_rate, prior_sampling_rate):
    
    ratio = target_sampling_rate / prior_sampling_rate
    
    duplication_indexes = np.arange(0, len(front_gps_data), 1)
    duplication_indexes = np.repeat(duplication_indexes, ratio)
    
    utm_lattitudes = np.array(utm_lattitudes)
    utm_lattitudes = np.repeat(utm_lattitudes, ratio, axis=0)

    utm_longitudes = np.array(utm_longitudes)
    utm_longitudes = np.repeat(utm_longitudes, ratio, axis=0)
    
    return duplication_indexes, utm_lattitudes, utm_longitudes

def merge_lattitudes_and_longitudes_in_complete_data(complete_data, utm_lattitudes, utm_longitudes):
    
    complete_data['utm_lat'] = utm_lattitudes
    complete_data['utm_lon'] = utm_longitudes
    
    return complete_data

def cut_utm_lattitudes_and_longitudes(duplication_indexes, utm_lattitudes, utm_longitudes, target_length):
    
    utm_lattitudes = utm_lattitudes[:target_length]
    utm_longitudes = utm_longitudes[:target_length]
    duplication_indexes = duplication_indexes[:target_length]
    
    return duplication_indexes, utm_lattitudes, utm_longitudes

def convert_timestamp_to_unix_ros_time(timestamp):
    return timestamp.apply(lambda x: x.timestamp())

def compute_common_unix_ros_time_elements(complete_data, gps_data):
    
    complete_data_ros_time_array = complete_data.ros_time.values
    gps_data_ros_time_array = gps_data.ros_time.values
    
    common_ros_time_array = np.intersect1d(complete_data_ros_time_array, gps_data_ros_time_array)
    
    return common_ros_time_array

def keep_data_with_same_unix_ros_time_elements(complete_data, gps_data, common_unix_ros_time_array):
    
    complete_data = complete_data[complete_data.ros_time.isin(common_unix_ros_time_array)]
    gps_data = gps_data[gps_data.ros_time.isin(common_unix_ros_time_array)]
    
    return complete_data, gps_data

def convert_complete_data_ros_time_from_nanoseconds_to_seconds(complete_data):
    complete_data.ros_time = complete_data.ros_time / 1e9
    return complete_data.ros_time

def get_absolute_angle_in_degrees_between_two_utm_coordinates(x1, y1, x2, y2):
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

def compute_angle_between_two_gps_data_arrays(utm_lattitudes_1, utm_longitudes_1, utm_lattitudes_2, utm_longitudes_2):
    
    angle = []
    
    for i in range(len(utm_lattitudes_1)):
        angle.append(get_absolute_angle_in_degrees_between_two_utm_coordinates(utm_lattitudes_1[i], utm_longitudes_1[i], utm_lattitudes_2[i], utm_longitudes_2[i]))
    
    return angle

def plot_angles(angles):
    plt.scatter(range(len(angles)), angles)
    plt.show()

def superpose_quiver_with_angles_on_utm_coordinates(utm_lattitudes, utm_longitudes, angles):
    
    plt.quiver(utm_lattitudes, utm_longitudes, np.cos(angles), np.sin(angles), angles, scale_units='xy', scale=0.5)
    plt.show()

if __name__ == "__main__":


    complete_data = pd.read_pickle('MS/GPS Pipeline/doughnut1_full_pickle.csv')
    complete_data['ros_time']  = convert_complete_data_ros_time_from_nanoseconds_to_seconds(complete_data).astype(int)

    # Import GPS data
    front_gps_data = import_gps_data_from_pos_file('MS/GPS Pipeline/data_september8th/reach-raw_202209081407.pos')
    middle_gps_data = import_gps_data_from_pos_file('MS/GPS Pipeline/data_september8th/reach_raw_202209081406.pos')
    back_gps_data = import_gps_data_from_pos_file('MS/GPS Pipeline/data_september8th/raw_202209081407.pos')
    

    # Convert timestamps to unix ros time
    front_gps_data['ros_time'] = convert_timestamp_to_unix_ros_time(front_gps_data.Timestamp).astype(int)
    middle_gps_data['ros_time'] = convert_timestamp_to_unix_ros_time(middle_gps_data.Timestamp).astype(int)
    back_gps_data['ros_time'] = convert_timestamp_to_unix_ros_time(back_gps_data.Timestamp).astype(int)

    # Keep common data between data sources

    common_unix_ros_time_array = compute_common_unix_ros_time_elements(front_gps_data, back_gps_data)
    front_gps_data, back_gps_data = keep_data_with_same_unix_ros_time_elements(front_gps_data, back_gps_data, common_unix_ros_time_array)
    common_unix_ros_time_array = compute_common_unix_ros_time_elements(front_gps_data, middle_gps_data)
    front_gps_data, middle_gps_data = keep_data_with_same_unix_ros_time_elements(front_gps_data, middle_gps_data, common_unix_ros_time_array)

    # Convert GPS data to UTM coordinates
    front_utm_lattitudes, front_utm_longitudes = get_utm_lattitudes_and_longitudes_from_gps_data(front_gps_data)
    middle_utm_lattitudes, middle_utm_longitudes = get_utm_lattitudes_and_longitudes_from_gps_data(middle_gps_data)
    back_utm_lattitudes, back_utm_longitudes = get_utm_lattitudes_and_longitudes_from_gps_data(back_gps_data)

    # TODO
    #duplication_indexes, front_utm_lattitudes, front_utm_longitudes = resample_utm_lattitudes_and_longitudes(front_utm_lattitudes, front_utm_longitudes, 20, 5)
    #duplication_indexes, front_utm_lattitudes, front_utm_longitudes = cut_utm_lattitudes_and_longitudes(duplication_indexes, front_utm_lattitudes, front_utm_longitudes, len(complete_data))
    
    # Compute angles between two GPS data arrays
    angles = compute_angle_between_two_gps_data_arrays(front_utm_lattitudes, front_utm_longitudes, back_utm_lattitudes, back_utm_longitudes)
    #plot_angles(angles)
    superpose_quiver_with_angles_on_utm_coordinates(front_utm_lattitudes, front_utm_longitudes, angles)

    plot_utm_coordinates(front_utm_lattitudes, front_utm_longitudes)
    plot_utm_coordinates(middle_utm_lattitudes, middle_utm_longitudes)
    plot_utm_coordinates(back_utm_lattitudes, back_utm_longitudes)
    
    # Merge GPS data with complete data
    #merged_data =  merge_gps_data_in_complete_data(complete_data, utm_lat, utm_lon)
    #pd.to_pickle(merged_data, 'MS/GPS Pipeline/grass_1_full_with_positions.csv')

    print('ALLO')




