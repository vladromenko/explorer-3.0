"""Explicit simulator allowlist; real profile names never instantiate a simulator."""
from .environment import SO101PickPlace
from .rosmaster_arm import RosmasterPickPlace
from .expert import ScriptedSO101,ScriptedRosmaster

ENVIRONMENTS={'so101_pick_place_sim':SO101PickPlace,'rosmaster_pick_place_sim':RosmasterPickPlace}
EXPERTS={'so101_pick_place_sim':ScriptedSO101,'rosmaster_pick_place_sim':ScriptedRosmaster}


def make(name,**kwargs):
    if name not in ENVIRONMENTS:raise ValueError('Unsupported simulation profile')
    return ENVIRONMENTS[name](**kwargs)
