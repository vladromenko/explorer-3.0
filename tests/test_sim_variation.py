import numpy as np
from explorer3.sim.factory import make,EXPERTS
from explorer3.sim.contact_audit import ContactAudit


def test_variations_repeat_and_restore_model_parameters_and_physics():
    env=make('so101_pick_place_sim',variation='varied')
    try:
        env.reset(seed=17);snapshot=env.snapshot();parameters=env.episode_parameters
        action=env.neutral_action()
        trajectory=[env.step(action)[0]['state'].copy() for _ in range(12)]
        env.reset(seed=18)
        assert parameters!=env.episode_parameters
        env.restore(snapshot)
        assert parameters==env.episode_parameters
        assert np.array_equal(trajectory,[env.step(action)[0]['state'].copy() for _ in range(12)])
        env.reset(seed=17,options={'variation':'nominal'})
        assert env.episode_parameters['mode']=='nominal'
        for key,value in env.variation.base.items():assert np.array_equal(getattr(env.model,key),value)
    finally:env.close()


def test_varied_contacts_still_hold_until_release_without_learning():
    for name in ('so101_pick_place_sim','rosmaster_pick_place_sim'):
        env=make(name,variation='varied')
        try:
            env.reset(seed=7);expert=EXPERTS[name](env)
            audit=ContactAudit(env.model);env.physics_observer=audit
            for _ in range(round(env.fps*env.deadline_s)+1):
                _,_,end,timeout,info=env.step(expert.action())
                if end or timeout:break
            assert info['success'] and audit.lifted and audit.minimum_carry_height_m>.045
            assert audit.minimum_distance_m>-.0005,audit.report()
        finally:env.close()
