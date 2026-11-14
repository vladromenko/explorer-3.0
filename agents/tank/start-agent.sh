#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1
# Camera and native ROS nodes are on this robot; only the HTTP agent uses Wi-Fi.
# Keep the existing ROS domain while avoiding raw-image DDS traffic on the WLAN.
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
export LD_LIBRARY_PATH="/home/ubuntu/explorer3/vendor/camera/lib:${LD_LIBRARY_PATH:-}"
cd /home/ubuntu/explorer3
exec .venv/bin/python -m uvicorn agent:app --host 0.0.0.0 --port 8041 --timeout-graceful-shutdown 5
