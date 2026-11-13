#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=0
cd /home/ubuntu/explorer3
exec .venv/bin/python sensors.py --localhost "$1"
