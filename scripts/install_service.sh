#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
install -m 644 scripts/explorer3-ui.service "$HOME/.config/systemd/user/explorer3-ui.service"
systemctl --user daemon-reload
./start.sh
