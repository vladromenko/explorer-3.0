"""M0: measure simulation throughput on Homebrain for every task and backend.

CPU: control steps per second of the existing gymnasium environments (1 process).
Warp: physics steps per second of the same MjModel batched on the Orin GPU.
Output: validation/sim-benchmark-<date>.json and a table on stdout.

Usage: MUJOCO_GL=egl PYTHONPATH=. .venv-sim/bin/python scripts/sim_benchmark.py [--quick]
"""
import json
import sys
import time
from datetime import date

import mujoco
import numpy as np

from explorer3.paths import DATA
from explorer3.sim.factory import make

TASKS = ['tank_dock_sim', 'rosmaster_dock_sim', 'so101_pick_place_sim', 'rosmaster_pick_place_sim']


def cpu_rate(name, seconds):
    env = make(name)
    env.reset(seed=1)
    action = np.asarray(env.neutral_action(), dtype=np.float32)
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    substeps = round(env.dt / env.model.opt.timestep)
    env.close()
    return {'control_steps_per_s': steps / elapsed, 'physics_steps_per_s': steps * substeps / elapsed,
            'control_hz': 1 / env.dt, 'substeps': substeps, 'realtime_factor': steps * env.dt / elapsed}


def warp_rate(model, nworld, steps):
    import warp as wp
    import mujoco_warp as mjw
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=nworld)
        mjw.step(m, d)  # compile
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            mjw.step(m, d)
        wp.synchronize()
        elapsed = time.perf_counter() - start
        return {'nworld': nworld, 'physics_steps_per_s': nworld * steps / elapsed, 'ok': True}
    except Exception as exc:  # report unsupported features instead of hiding them
        return {'nworld': nworld, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:300]}


def main():
    quick = '--quick' in sys.argv
    results = {'date': str(date.today()), 'machine': 'Homebrain Jetson Orin NX 16GB', 'mujoco': mujoco.__version__,
               'tasks': {}}
    for name in TASKS:
        row = {'cpu': cpu_rate(name, 3 if quick else 10)}
        env = make(name)
        row['warp'] = [warp_rate(env.model, n, 20 if quick else 100) for n in ((64,) if quick else (64, 256, 1024))]
        row['model'] = {'nq': env.model.nq, 'nv': env.model.nv, 'ngeom': env.model.ngeom,
                        'nmesh': env.model.nmesh, 'timestep': env.model.opt.timestep}
        env.close()
        results['tasks'][name] = row
        cpu = row['cpu']
        print(f"{name:26s} CPU {cpu['control_steps_per_s']:8.1f} ctrl/s ({cpu['realtime_factor']:5.1f}x realtime)", end='')
        for w in row['warp']:
            print(f" | warp n={w['nworld']}: " + (f"{w['physics_steps_per_s']:,.0f} phys/s" if w['ok'] else 'FAIL ' + w['error'][:60]), end='')
        print()
    out = DATA / f'validation/sim-benchmark-{date.today()}.json'
    out.write_text(json.dumps(results, indent=1))
    print('saved', out)


if __name__ == '__main__':
    main()
