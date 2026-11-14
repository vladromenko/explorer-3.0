"""Compatibility facade: all hardware admission comes from fleet.json."""
from .paths import DATA
from . import fleet_config


def homebrain_only():
    try:return not any(r['enabled'] for r in fleet_config.load(DATA)['robots'].values())
    except ValueError:return True


def workflow_stage():
    # Stage labels are not another authorization layer.
    return None


def enabled():return homebrain_only()


def robot_exception(robot):return hardware_allowed(robot)


def hardware_allowed(robot=None):return fleet_config.robot_allowed(robot,DATA)[0]


def require_hardware_mode(robot=None):fleet_config.require_robot(robot,DATA)
