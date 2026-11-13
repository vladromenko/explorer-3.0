#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sim_data="${EXPLORER3_DATA:-$HOME/.local/share/explorer3}"
python3 -m venv .venv-sim
.venv-sim/bin/python -m pip install -r configs/sim-requirements.lock.txt
source_dir="$sim_data/cache/source/mujoco_menagerie"
if [ ! -d "$source_dir/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$source_dir"
fi
git -C "$source_dir" sparse-checkout set robotstudio_so101
git -C "$source_dir" checkout 370c43965323c0944b433b202aac5058b5f907a9
PLAYWRIGHT_BROWSERS_PATH="$sim_data/cache/playwright" .venv-sim/bin/python -m playwright install chromium
.venv-sim/bin/python -m pip check
