#!/usr/bin/env bash
# Run the ROSMASTER agent tests ON the robot against the copied FakeBoard (no motors, no serial).
# Usage: scripts/rosmaster_robot_tests.sh   (uses ~/.ssh/rosmaster-collector and the agent config host)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
DATA="${EXPLORER3_DATA:-$HOME/.local/share/explorer3}"
HOST=$(python3 -c "import json,sys;print(json.load(open('$DATA/configs/rosmaster-agent.json'))['host'])")
KEY="$HOME/.ssh/rosmaster-collector"; R="ssh -i $KEY -o BatchMode=yes vlad@$HOST"
SRC="$DATA/provenance/rosmaster-bridge-current"
$R 'rm -rf /tmp/e3test && mkdir -p /tmp/e3test/fake/rosmaster /tmp/e3test/agents/rosmaster && touch /tmp/e3test/agents/__init__.py /tmp/e3test/agents/rosmaster/__init__.py && printf "import rosmaster\nrosmaster.__path__.append(\"/tmp/e3test/fake/rosmaster\")\n" > /tmp/e3test/conftest.py'
scp -q -i "$KEY" "$SRC/CubeBot/rosmaster/fake_board.py" "vlad@$HOST:/tmp/e3test/fake/rosmaster/"
scp -q -i "$KEY" "$SRC/explorer3/config/controller-calibration.json" "vlad@$HOST:/tmp/e3test/calibration.json"
scp -q -i "$KEY" agents/rosmaster/{runtime.py,leased_core.py,control_adapter.py,agent.py,handoff.py,test_runtime.py} agents/rosmaster/console-fixes/test_explorer_adapter.py "vlad@$HOST:/tmp/e3test/"
scp -q -i "$KEY" agents/rosmaster/{control_adapter.py,control-adapter-provenance.json} "vlad@$HOST:/tmp/e3test/agents/rosmaster/"
$R 'source ~/explorer3/environment.sh && cd ~/explorer3 && export PYTHONPATH=/tmp/e3test:$PYTHONPATH EXPLORER3_TEST_CALIBRATION=/tmp/e3test/calibration.json && timeout 300 ~/explorer3/.venv/bin/python -m pytest -q -p no:cacheprovider /tmp/e3test/test_runtime.py /tmp/e3test/test_explorer_adapter.py'
