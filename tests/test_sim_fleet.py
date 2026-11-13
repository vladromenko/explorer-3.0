"""Actual shared-world dynamics; no optimization and no hardware."""
import copy
import numpy as np
from explorer3.sim.fleet import FleetWorld
from explorer3.fleet_allocation import FleetAllocator


def test_shared_dynamics_complete_and_partial_reset_preserves_other_robots():
    world=FleetWorld(seed=1)
    try:
        world.demonstrate()
        for _ in range(330):world.step()
        assert all(v.done and v.result[4]['outcome']=='success' for v in world.views.values())
        before=world.data.qpos.copy();velocity=world.data.qvel.copy();clock=world.data.time
        ids={k:v.task_id for k,v in world.views.items()}
        world.reset(seed=32,agent='tank')
        for name in ('so101','rosmaster'):
            view=world.views[name]
            np.testing.assert_array_equal(world.data.qpos[view.q],before[view.q])
            np.testing.assert_array_equal(world.data.qvel[view.v],velocity[view.v])
            assert view.task_id==ids[name]
        assert world.data.time==clock and world.views['tank'].task_id==ids['tank']+1
    finally:world.close()


def test_snapshot_includes_pending_targets_experts_and_exclusive_scheduler():
    world=FleetWorld(seed=2)
    try:
        world.demonstrate()
        for _ in range(93):world.step()
        allocator=FleetAllocator(3,'fleet_thrifty',seed=12,min_steps=2)
        for _ in range(6):
            chosen=allocator.allocate([.8,.9,.1],[.7,.8,.1])
            world.assign_operator(list(world.views)[chosen] if chosen is not None else None)
            assert allocator.assignments.sum()<=1
        world.scheduler_state=allocator.snapshot()
        saved=world.snapshot()
        for _ in range(16):world.step()
        expected=world.data.qpos.copy();vel=world.data.qvel.copy();tasks=copy.deepcopy(world.info()['robots'])
        world.restore(saved)
        restored=FleetAllocator(3,'fleet_thrifty',seed=12,min_steps=2);restored.restore(world.scheduler_state)
        assert allocator.allocate([.4,.8,.2],[.3,.9,.1])==restored.allocate([.4,.8,.2],[.3,.9,.1])
        for _ in range(16):world.step()
        np.testing.assert_allclose(world.data.qpos,expected,atol=1e-10,rtol=0)
        np.testing.assert_allclose(world.data.qvel,vel,atol=1e-9,rtol=0)
        assert world.info()['robots']==tasks
    finally:world.close()
