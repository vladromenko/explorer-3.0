#!/bin/bash
source /opt/ros/jazzy/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=30 ROS_LOCALHOST_ONLY=1
export EXPLORER3_ROOT=/home/vlad/explorer3
export EXPLORER_ROOT="$EXPLORER3_ROOT"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$EXPLORER3_ROOT/config/fastdds-rgbd.xml"
for prefix in "$EXPLORER3_ROOT"/vendor/ros/*; do
  export AMENT_PREFIX_PATH="$prefix:${AMENT_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$prefix/lib/python3.12/site-packages:${PYTHONPATH:-}"
done
export PYTHONPATH="$EXPLORER3_ROOT/native:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
