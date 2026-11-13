#!/bin/bash
set -e
source /home/vlad/explorer3/environment.sh
cd "$EXPLORER3_ROOT"
exec .venv/bin/python sensors.py --robot rosmaster --localhost "$1" --port "$((8053 + $1))"
