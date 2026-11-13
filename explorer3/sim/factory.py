"""Explicit simulator allowlist; real profile names never instantiate a simulator."""
from .environment import SO101PickPlace
from .rosmaster_arm import RosmasterPickPlace
from .expert import ScriptedSO101,ScriptedRosmaster
from .mobile import MobileDock,DockExpert
from functools import partial

ENVIRONMENTS={'so101_pick_place_sim':SO101PickPlace,'rosmaster_pick_place_sim':RosmasterPickPlace,
              'rosmaster_dock_sim':partial(MobileDock,'rosmaster'),'tank_dock_sim':partial(MobileDock,'tank')}
EXPERTS={'so101_pick_place_sim':ScriptedSO101,'rosmaster_pick_place_sim':ScriptedRosmaster,'rosmaster_dock_sim':DockExpert,'tank_dock_sim':DockExpert}


def make(name,**kwargs):
    if name not in ENVIRONMENTS:raise ValueError('Unsupported simulation profile')
    return ENVIRONMENTS[name](**kwargs)
