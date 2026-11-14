"""M0: mujoco_warp throughput with CUDA graph capture. Usage: MUJOCO_GL=egl PYTHONPATH=. .venv-sim/bin/python scripts/sim_benchmark_warp_graph.py <task> <nworld>..."""
import time,sys,mujoco,numpy as np,warp as wp,mujoco_warp as mjw
from explorer3.sim.factory import make
name=sys.argv[1]
env=make(name);model=env.model;data=mujoco.MjData(model);mujoco.mj_forward(model,data)
m=mjw.put_model(model)
for n in map(int,sys.argv[2:]):
    d=mjw.put_data(model,data,nworld=n)
    mjw.step(m,d);wp.synchronize()
    with wp.ScopedCapture() as cap: 
        for _ in range(10): mjw.step(m,d)
    wp.capture_launch(cap.graph);wp.synchronize()
    t=time.perf_counter()
    for _ in range(10): wp.capture_launch(cap.graph)
    wp.synchronize();e=time.perf_counter()-t
    print(name,n,'graph phys/s',int(n*100/e),flush=True)
