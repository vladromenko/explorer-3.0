import pytest
from explorer3.sim.ownership import OperatorLease


def test_stop_and_expiry_require_explicit_reacquisition():
    time=[0.];lease=OperatorLease(clock=lambda:time[0])
    lease.claim('first-operator');lease.require('first-operator')
    with pytest.raises(ValueError):lease.claim('second-operator')
    lease.release('second-operator');assert lease.owner=='first-operator'
    time[0]=3;assert lease.expire()
    with pytest.raises(ValueError):lease.require('first-operator')
    lease.claim('second-operator');lease.release(force=True)
    with pytest.raises(ValueError):lease.require('second-operator')


def test_mobile_force_response_and_snapshot():
    import numpy as np
    from explorer3.sim.mobile import MobileDock
    for robot in ('tank','rosmaster'):
        env=MobileDock(robot);before=env.data.qpos.copy()
        target=[.2,.2] if robot=='tank' else [0,.2,0]
        snapshot=env.snapshot();env.step(target)
        assert np.array_equal(before,env.data.qpos)
        path=[env.step(target)[0]['state'].copy() for _ in range(15)]
        assert np.linalg.norm(env.data.qpos-before)>.05
        env.restore(snapshot);env.step(target)
        assert np.array_equal(path,[env.step(target)[0]['state'].copy() for _ in range(15)])
        if robot=='tank':assert env.observation()['state'].shape==(2,)
        for _ in range(30):env.step(env.neutral_action())
        assert np.linalg.norm(env.data.qvel)<.001
        env.close()
