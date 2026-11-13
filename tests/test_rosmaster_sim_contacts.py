"""Actual original geometry: contacts measured at every physics substep."""
import pytest
from explorer3.sim.factory import make,EXPERTS
from explorer3.sim.contact_audit import ContactAudit


@pytest.mark.parametrize('profile',['so101_pick_place_sim','rosmaster_pick_place_sim'])
def test_native_gripper_holds_and_releases_without_sinking(profile):
    env=make(profile)
    try:
        for seed in (1,2,3):
            env.reset(seed=seed);expert=EXPERTS[profile](env)
            audit=ContactAudit(env.model);env.physics_observer=audit
            for _ in range(round(env.fps*env.deadline_s)+1):
                _,_,terminated,truncated,info=env.step(expert.action())
                if terminated or truncated:break
            assert audit.lifted and info['success'] and info['released'] and info['inside_box']
            assert audit.minimum_distance_m>-.0005, audit.report()
            assert audit.maximum_mimic_error_rad<.001, audit.report()
            assert audit.minimum_carry_height_m>.045, 'Dropped before commanded release: '+str(audit.report())
    finally:env.close()
