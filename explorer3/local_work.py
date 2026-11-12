"""Persistent local-only mode: no remote probes or physical resource acquisition."""
import json
from .paths import DATA


def enabled():
    path = DATA/'configs/local-only.json'
    return path.exists() and json.loads(path.read_text()).get('enabled') is True


def require_hardware_mode():
    if enabled():
        raise PermissionError('Включён локальный режим: роботы не опрашиваются, управление не открывается')
