#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 -m venv .venv
.venv/bin/pip install 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r configs/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip check
