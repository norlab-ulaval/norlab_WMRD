#! /bin/bash

if [ $# -ne 2 ]
then
  echo "Incorrect number of arguments! Argument 1 is the folder containing the bag files. Argument 2 is the folder in which to store the results."
  exit
fi

bag_path=$1
results_path=$2

csv_file=$results_path"/data".csv
traj_file=$results_path"/traj".vtk
map_file=$results_path"/map".vtk

#echo $csv_file

## start nodes and pause playing
roslaunch norlab_imu_tools husky_imu_and_wheel_odom.launch&
roslaunch husky_mapping realtime_mapping.launch&
roslaunch pose_cmds_logger logger.launch &
roslaunch norlab_bag_player husky.launch bagfile:=$bag_path rate:=0.5&
rviz &
sleep 5
rosservice call /play/pause_playback "data: false"

## wait until the end of the bag
while [[ ! -z `pgrep play` ]]
do
sleep 1
done

## save the map
sleep 10
rosservice call /save_map "map_file_name:
#data: '$map_file'"
rosservice call /save_trajectory "trajectory_file_name:
data: '$trajectory_file'"
rosservice call /save_data "trajectory_file_name:
data: '$data_file'"
#
## kill everything
killall rviz
killall pointcloud2_deskew_node
killall mapper_node
killall cloud_node_stamped
killall rosmaster
