"""Persistent local-only mode: no remote probes or physical resource acquisition."""
import json
from .paths import DATA


def enabled():
    path = DATA/'configs/local-only.json'
    return path.exists() and json.loads(path.read_text()).get('enabled') is True


def hardware_allowed(robot=None):
    if not enabled():return True
    if robot=='tank':
        path=DATA/'configs/tank-hardware-enabled.json'
        return path.exists() and json.loads(path.read_text()).get('enabled') is True
    return False


def require_hardware_mode(robot=None):
    if not hardware_allowed(robot):
        raise PermissionError('Включён локальный режим: роботы не опрашиваются, управление не открывается')
