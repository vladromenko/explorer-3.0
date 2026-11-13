"""Persistent local-only mode: no remote probes or physical resource acquisition."""
import json
from .paths import DATA


def homebrain_only():
    path=DATA/'configs/workflow.json'
    return path.exists() and json.loads(path.read_text()).get('stage')=='A'


def workflow_stage():
    path=DATA/'configs/workflow.json'
    if not path.exists():return None
    value=json.loads(path.read_text()).get('stage')
    return value if value in ('A','B','C','D','E','F') else None


def enabled():
    path = DATA/'configs/local-only.json'
    return path.exists() and json.loads(path.read_text()).get('enabled') is True


def robot_exception(robot):
    """Operator-agreed hardware exception for exactly one robot while the global stage keeps others closed."""
    path=DATA/'configs'/f'{robot}-hardware-enabled.json'
    return robot=='rosmaster' and path.exists() and json.loads(path.read_text()).get('enabled') is True


def hardware_allowed(robot=None):
    path=DATA/'configs/workflow.json'
    workflow=json.loads(path.read_text()) if path.exists() else {}
    # Current-stage scope wins over stale authorization files from older stages.
    if 'allowed_robots' in workflow:
        allowed=workflow['allowed_robots']
        if not isinstance(allowed,list) or robot not in allowed:return False
    if robot_exception(robot):return True
    if homebrain_only():return False
    if not enabled():return True
    if robot=='tank':
        path=DATA/'configs/tank-hardware-enabled.json'
        return path.exists() and json.loads(path.read_text()).get('enabled') is True
    return False


def require_hardware_mode(robot=None):
    if not hardware_allowed(robot):
        raise PermissionError('Включён локальный режим: роботы не опрашиваются, управление не открывается')
