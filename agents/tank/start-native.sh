#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/ubuntu/ros2_ws/install/ros_robot_controller_msgs/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/ubuntu/ros2_ws/install/ros_robot_controller_msgs/lib:${LD_LIBRARY_PATH:-}"
cd /home/ubuntu/explorer3
exec .venv/bin/python native_driver.py "$@" --ros-args -r __ns:=/explorer3/tank -p pub_odom_topic:=false -r /app/cmd_vel:=/explorer3/tank/unused_app_cmd_vel
