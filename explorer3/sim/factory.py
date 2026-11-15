"""Explicit simulator allowlist; real profile names never instantiate a simulator."""
from .environment import SO101PickPlace
from .rosmaster_arm import RosmasterPickPlace
from .expert import ScriptedSO101,ScriptedRosmaster
from .mobile import MobileDock,DockExpert
from functools import partial
from .aligned import SO101Handover, RosmasterFloorPick, ScriptedHandover, ScriptedFloorPick

ENVIRONMENTS={'so101_pick_place_sim':SO101PickPlace,'rosmaster_pick_place_sim':RosmasterPickPlace,
              'rosmaster_dock_sim':partial(MobileDock,'rosmaster'),'tank_dock_sim':partial(MobileDock,'tank'),
              'so101_handover_sim':SO101Handover,'rosmaster_floor_pick_sim':RosmasterFloorPick,
              # knowledge-gap root: varied objects; the task text carries the SO-101 measurement or not
              'rosmaster_floor_pick_knowledge_sim':partial(RosmasterFloorPick,vary_objects=True,with_knowledge=True),
              'rosmaster_floor_pick_noknowledge_sim':partial(RosmasterFloorPick,vary_objects=True,with_knowledge=False)}
EXPERTS={'so101_pick_place_sim':ScriptedSO101,'rosmaster_pick_place_sim':ScriptedRosmaster,'rosmaster_dock_sim':DockExpert,'tank_dock_sim':DockExpert,
         'so101_handover_sim':ScriptedHandover,'rosmaster_floor_pick_sim':ScriptedFloorPick,
         'rosmaster_floor_pick_knowledge_sim':ScriptedFloorPick,'rosmaster_floor_pick_noknowledge_sim':ScriptedFloorPick}


def make(name,**kwargs):
    if name not in ENVIRONMENTS:raise ValueError('Unsupported simulation profile')
    return ENVIRONMENTS[name](**kwargs)
