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

# start nodes and pause playing
roscore &
rosbag play bag_path rate:=0.5 --clock &
sleep 5
rosservice call /play/pause_playback "data: true"
roslaunch husky_mapping realtime_mapping.launch csv_file_name:=$csv_file &
roslaunch pose_cmd_logger logger.launch &
rviz &
sleep 5
rosservice call /play/pause_playback "data: false"

# wait until the end of the bag
while [[ ! -z `pgrep play` ]]
do
sleep 1
done

# save the map
sleep 10
rosservice call /save_map "map_file_name:
data: '$map_file'"
rosservice call /save_trajectory "trajectory_file_name:
data: '$trajectory_file'"
rosservice call /save_data "trajectory_file_name:
data: '$data_file'"

# kill everything
killall rviz
killall pointcloud2_deskew_node
killall mapper_node
killall cloud_node_stamped
killall rosmaster
