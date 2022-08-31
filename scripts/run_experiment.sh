#! /bin/bash

if [ $# -ne 2 ]
then
  echo "Incorrect number of arguments! Argument 1 is the folder containing the bag files. Argument 2 is the folder in which to store the results."
  exit
fi

bag_path=$1
results_path=$2

csv_file="$results_path"/data.csv
traj_file="$results_path"/traj.vtk
map_file="$results_path"/map.vtk

echo $csv_file &
echo $traj_file &
echo $map_file &

## start nodes and pause playing
#sleep 5 | yes | rosclean purge &
roslaunch norlab_bag_player husky.launch bagfile:=$bag_path rate:=0.5 &
sleep 10
#roslaunch norlab_imu_tools husky_imu_and_wheel_odom.launch &
roslaunch husky_mapping realtime_mapping.launch &
roslaunch pose_cmds_logger logger.launch &
rviz &
sleep 5
rosservice call /play/pause_playback "data: false"

### wait until the end of the bag
while [[ ! -z `pgrep play` ]]
do
sleep 1
done

## save the map and data
sleep 5
rosservice call /save_map "map_file_name:
 data: '$map_file'"
sleep 20
rosservice call /save_trajectory "trajectory_file_name:
 data: '$traj_file'"
sleep 5
rosservice call /save_data "data_file_name:
 data: '$csv_file'"

#
## kill everything
sleep 30
killall rviz
killall pcl_deskew_node
killall imu_and_wheel_odom_node
killall husky_dataset_logger
killall mapper_node
killall map_throttler
killall cloud_node_stamped
killall rosmaster
