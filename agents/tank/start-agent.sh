#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/home/ubuntu/explorer3/vendor/camera/lib:${LD_LIBRARY_PATH:-}"
cd /home/ubuntu/explorer3
exec .venv/bin/python -m uvicorn agent:app --host 0.0.0.0 --port 8041 --timeout-graceful-shutdown 5
