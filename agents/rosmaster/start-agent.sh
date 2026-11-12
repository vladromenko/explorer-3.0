#!/bin/bash
set -e
source /home/vlad/explorer3/environment.sh
cd "$EXPLORER3_ROOT"
exec .venv/bin/python -m uvicorn agent:app --host 0.0.0.0 --port 8042 --timeout-graceful-shutdown 5
