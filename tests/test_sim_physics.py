"""Run with .venv-sim; actual dynamics and branch determinism, no fake simulator."""
import numpy as np
from explorer3.sim.environment import SO101PickPlace


def test_delayed_physics_and_complete_restore():
    env=SO101PickPlace()
    obs,_=env.reset(seed=12)
    target=obs['state'].copy();target[0]+=15
    snapshot=env.snapshot()
    initial=env.data.qpos.copy()
    env.step(target)
    assert abs(env.data.qpos[0]-initial[0])<.01  # transport delay, no qpos teleport
    states=[]
    for _ in range(12):
        states.append(env.step(target)[0]['state'].copy())
    assert abs(env.data.qpos[0]-initial[0])>.03
    env.restore(snapshot);env.step(target)
    repeated=[env.step(target)[0]['state'].copy() for _ in range(12)]
    assert np.array_equal(states,repeated)
    env.close()


def test_deadline_is_scientific_failure_and_observation_has_no_truth():
    env=SO101PickPlace(deadline_s=.1)
    obs,_=env.reset(seed=21)
    assert set(obs)=={'state'}
    finished=False
    while not finished:
        _,_,terminated,truncated,info=env.step(obs['state'])
        finished=terminated or truncated
    assert truncated and info['outcome']=='failure' and info['termination_reason']=='task_deadline'
    env.close()
