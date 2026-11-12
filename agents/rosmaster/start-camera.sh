#!/bin/bash
set -e
source /home/vlad/explorer3/environment.sh
exec ros2 launch "$EXPLORER3_ROOT/control/camera-rgbd.launch.py" enable_point_cloud:=false enable_ir:=false depth_registration:=true color_fps:=15 depth_fps:=15 enable_frame_sync:=false frame_aggregate_mode:=disable
