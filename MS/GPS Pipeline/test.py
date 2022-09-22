import pandas as pd
import utm
import matplotlib.pyplot as plt
import numpy as np






########################################################################################
# Methods. Ordered by execution order
########################################################################################
def import_gps_data_from_pos_file(csv_file_path):
    gps_data = pd.read_table(csv_file_path, sep="\s+", parse_dates={"Timestamp": [0, 1]}, error_bad_lines=False, skiprows=25)
    return gps_data

def convert_complete_data_unix_ros_time_from_nanoseconds_to_seconds(complete_data):
    complete_data.ros_time = complete_data.ros_time / 1e9
    return complete_data.ros_time.astype(int)

def convert_timestamp_to_unix_ros_time(timestamp):
    return timestamp.apply(lambda x: x.timestamp()).astype(int)

def keep_common_unix_ros_time_elements_in_both_complete_data_and_single_gps_data(complete_data, single_gps_data):
    common_unix_ros_time_array = np.intersect1d(complete_data.ros_time.values, single_gps_data.ros_time.values)
    complete_data = complete_data[complete_data.ros_time.isin(common_unix_ros_time_array)]
    single_gps_data = single_gps_data[single_gps_data.ros_time.isin(common_unix_ros_time_array)]
    return complete_data, single_gps_data

def get_utm_latitudes_and_longitudes_from_single_gps_data(gps_data):
    lattitudes = gps_data.values[:,1].astype(float)
    longitudes = gps_data.values[:,2].astype(float)
    utm_lattitudes = []
    utm_longitudes = []
    for i in range(len(lattitudes)):
        utm_lattitudes.append(utm.from_latlon(lattitudes[i], longitudes[i])[0])
        utm_longitudes.append(utm.from_latlon(lattitudes[i], longitudes[i])[1])
    return utm_lattitudes, utm_longitudes

def resample_utm_latitudes_and_longitudes(utm_lattitudes, utm_longitudes, target_sampling_rate, prior_sampling_rate):

    ratio = target_sampling_rate / prior_sampling_rate
    
    duplication_indexes = np.arange(0, len(utm_lattitudes), 1)
    duplication_indexes = np.repeat(duplication_indexes, ratio)
    
    utm_lattitudes = np.array(utm_lattitudes)
    utm_lattitudes = np.repeat(utm_lattitudes, ratio, axis=0)

    utm_longitudes = np.array(utm_longitudes)
    utm_longitudes = np.repeat(utm_longitudes, ratio, axis=0)
    
    return duplication_indexes, utm_lattitudes, utm_longitudes

def cut_utm_lattitudes_and_longitudes(duplication_indexes, utm_lattitudes, utm_longitudes, target_length):  
    utm_lattitudes = utm_lattitudes[:target_length]
    utm_longitudes = utm_longitudes[:target_length]
    
    duplication_indexes = duplication_indexes[:target_length]
    
    return duplication_indexes, utm_lattitudes, utm_longitudes

def compute_angle_between_back_and_front_gps(utm_lattitudes_1, utm_longitudes_1, utm_lattitudes_2, utm_longitudes_2):
    
    angles = []
    hardcoded_angle_between_the_two_gps_installed_on_robot = 16.48
    
    for i in range(len(utm_lattitudes_1)):
        
        x1 = utm_lattitudes_1[i]
        x2 = utm_lattitudes_2[i]
        y1 = utm_longitudes_1[i]
        y2 = utm_longitudes_2[i]
        
        angle = (np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) - hardcoded_angle_between_the_two_gps_installed_on_robot
        angles.append(angle)
    
    return angles






########################################################################################
# Utils for visualization
########################################################################################
def plot_utm_coordinates(utm_lattitudes, utm_longitudes):
    plt.plot(utm_lattitudes, utm_longitudes, '.')
    plt.show()
def plot_angles(angles):
    plt.plot(range(len(angles)), angles)
    plt.show()
def superpose_quiver_with_angles_on_utm_coordinates(utm_lattitudes, utm_longitudes, angles):
    plt.figure()
    plt.quiver(utm_lattitudes, utm_longitudes, np.cos(np.radians(angles)), np.sin(np.radians(angles)), np.radians(angles), scale_units='xy', scale=1.5)
    plt.title('Quiver plot')
    plt.xlabel('Position X (meters)')
    plt.ylabel('Position Y (meters)')
    plt.show()






########################################################################################
# Main
########################################################################################
def main(complete_data_path, front_gps_data_path, middle_gps_data_path, back_gps_data_path, complete_data_sampling_rate, gps_data_sampling_rate):

    # Import data
    complete_data = pd.read_pickle(complete_data_path)
    front_gps_data = import_gps_data_from_pos_file(front_gps_data_path)
    middle_gps_data = import_gps_data_from_pos_file(middle_gps_data_path)
    back_gps_data = import_gps_data_from_pos_file(back_gps_data_path)
    
    # Uniformize timestamps
    complete_data['ros_time']  = convert_complete_data_unix_ros_time_from_nanoseconds_to_seconds(complete_data)
    front_gps_data['ros_time'] = convert_timestamp_to_unix_ros_time(front_gps_data.Timestamp)
    middle_gps_data['ros_time'] = convert_timestamp_to_unix_ros_time(middle_gps_data.Timestamp)
    back_gps_data['ros_time'] = convert_timestamp_to_unix_ros_time(back_gps_data.Timestamp)

    # Keep common data between data sources
    complete_data, front_gps_data = keep_common_unix_ros_time_elements_in_both_complete_data_and_single_gps_data(complete_data, front_gps_data)
    complete_data, middle_gps_data = keep_common_unix_ros_time_elements_in_both_complete_data_and_single_gps_data(complete_data, middle_gps_data)
    complete_data, back_gps_data = keep_common_unix_ros_time_elements_in_both_complete_data_and_single_gps_data(complete_data, back_gps_data)

    # Convert GPS data to UTM coordinates
    front_utm_latitudes, front_utm_longitudes = get_utm_latitudes_and_longitudes_from_single_gps_data(front_gps_data)
    middle_utm_latitudes, middle_utm_longitudes = get_utm_latitudes_and_longitudes_from_single_gps_data(middle_gps_data)
    back_utm_latitudes, back_utm_longitudes = get_utm_latitudes_and_longitudes_from_single_gps_data(back_gps_data)

    # Resample GPS data to match complete data sampling rate
    duplication_indexes, front_utm_latitudes, front_utm_longitudes = resample_utm_latitudes_and_longitudes(front_utm_latitudes, front_utm_longitudes, complete_data_sampling_rate, gps_data_sampling_rate)
    duplication_indexes, middle_utm_latitudes, middle_utm_longitudes = resample_utm_latitudes_and_longitudes(middle_utm_latitudes, middle_utm_longitudes, complete_data_sampling_rate, gps_data_sampling_rate)
    duplication_indexes, back_utm_latitudes, back_utm_longitudes = resample_utm_latitudes_and_longitudes(back_utm_latitudes, back_utm_longitudes, complete_data_sampling_rate, gps_data_sampling_rate)

    # Cut GPS data to match complete data length
    duplication_indexes, front_utm_latitudes, front_utm_longitudes = cut_utm_lattitudes_and_longitudes(duplication_indexes, front_utm_latitudes, front_utm_longitudes, len(complete_data))
    duplication_indexes, middle_utm_latitudes, middle_utm_longitudes = cut_utm_lattitudes_and_longitudes(duplication_indexes, middle_utm_latitudes, middle_utm_longitudes, len(complete_data))
    duplication_indexes, back_utm_latitudes, back_utm_longitudes = cut_utm_lattitudes_and_longitudes(duplication_indexes, back_utm_latitudes, back_utm_longitudes, len(complete_data))

    # Compute angles between two GPS data arrays
    angle_between_front_and_back_gps = compute_angle_between_back_and_front_gps(front_utm_latitudes, front_utm_longitudes, back_utm_latitudes, back_utm_longitudes)

    # Merge GPS data with complete data 
    complete_data['front_gps_utm_latitudes'] = front_utm_latitudes
    complete_data['front_gps_utm_longitudes'] = front_utm_longitudes
    complete_data['middle_gps_utm_latitudes'] = middle_utm_latitudes
    complete_data['middle_gps_utm_longitudes'] = middle_utm_longitudes
    complete_data['back_gps_utm_latitudes'] = back_utm_latitudes
    complete_data['back_gps_utm_longitudes'] = back_utm_longitudes
    complete_data['angle_between_front_and_back_gps'] = angle_between_front_and_back_gps
    complete_data['duplication_indexes'] = duplication_indexes
    
    # Plot angles
    # Uncomment the following lines to plot angles
    #plot_angles(angle_between_front_and_back_gps)
    #superpose_quiver_with_angles_on_utm_coordinates(front_utm_latitudes, front_utm_longitudes, angle_between_front_and_back_gps)
    #plot_utm_coordinates(front_utm_latitudes, front_utm_longitudes)
    #plot_utm_coordinates(middle_utm_latitudes, middle_utm_longitudes)
    #plot_utm_coordinates(back_utm_latitudes, back_utm_longitudes)


if __name__ == "__main__":

    complete_data_path = "MS/GPS Pipeline/data_september8th/doughnut__2022-09-08-10-11-57.pkl"
    front_gps_path = "MS/GPS Pipeline/data_september8th/reach-raw_202209081407.pos"
    middle_gps_path = "MS/GPS Pipeline/data_september8th/reach_raw_202209081406.pos"
    back_gps_path = "MS/GPS Pipeline/data_september8th/raw_202209081407.pos"
    new_complete_data_path = "MS/GPS Pipeline/grass_1_full_with_positions.pkl"
    complete_data_sampling_rate_in_hz = 20
    gps_data_sampling_rate_in_hz = 5

    complete_data = main(complete_data_path, front_gps_path, middle_gps_path, back_gps_path, complete_data_sampling_rate_in_hz, gps_data_sampling_rate_in_hz)
    pd.to_pickle(complete_data, new_complete_data_path)