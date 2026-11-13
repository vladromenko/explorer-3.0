"""Run on ROSMASTER's Python with candidate control_adapter on PYTHONPATH.

Uses only native geometry and recorded poses; creates no Runtime, serial or ROS
node. The executing directory must not shadow the candidate module.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cubebot import config
from cubebot.arm_model import ArmModel
from cubebot.arm_control import CartesianJog
import control_adapter

cfg=config.load();model=ArmModel(cfg['arm']['tcp_offset_m'])
timings=[];results=[]
for axis,sign in [(0,-1),(0,1),(2,-1),(2,1)]:
    pose=[88,76,93,90,90,45]
    jog=CartesianJog(control_adapter.ManualCartesianModel(model),cfg,pose)
    jog.WORKSPACE=((-np.inf,np.inf),)*3
    start=model.tcp_position(pose);move=[0,0,0];move[axis]=sign
    for _ in range(20):
        before=time.perf_counter();jog.step(.05,move,slow=True)
        timings.append((time.perf_counter()-before)*1000)
    results.append({'axis':axis,'sign':sign,'delta_mm':np.round((model.tcp_position(jog.servo)-start)*1000,3).tolist(),
                    'blocked':jog.blocked})
source=Path(control_adapter.__file__)
report={'hardware_access':False,'training_started':False,'candidate':str(source),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'steps':len(timings),
        'step_ms_p50_p95_p99_max':np.percentile(timings,[50,95,99,100]).round(3).tolist(),'directions':results}
print(json.dumps(report))
assert max(timings)<100,'XYZ calculation exceeds its 100 ms offline timing budget'
