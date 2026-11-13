"""Save/reload checks only; deliberately no learn() / optimizer steps."""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from explorer3.sim.environment import SO101PickPlace,StateTeacher
from explorer3.sim import checkpoints


def test_policy_and_frozen_normalization_roundtrip_without_training(tmp_path):
    env=VecNormalize(DummyVecEnv([lambda:StateTeacher(SO101PickPlace())]),norm_obs=True,norm_reward=True)
    other=None
    try:
        obs=env.reset()
        for _ in range(8):obs,_,_,_=env.step(np.zeros((1,6)))
        model=PPO('MlpPolicy',env,device='cpu',n_steps=8,batch_size=8,seed=17)
        assert model.num_timesteps==0
        path=checkpoints.publish(model,tmp_path)
        assert checkpoints.latest(tmp_path)==path
        other=VecNormalize.load(checkpoints.verify(path),DummyVecEnv([lambda:StateTeacher(SO101PickPlace())]))
        other.training=False;other.norm_reward=False
        loaded=PPO.load(path,env=other,device='cpu')
        raw=env.get_original_obs()
        assert np.array_equal(env.normalize_obs(raw),other.normalize_obs(raw))
        assert np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
        before=other.obs_rms.count;other.reset();other.step(np.zeros((1,6)))
        assert other.obs_rms.count==before and loaded.num_timesteps==0
    finally:
        env.close()
        if other is not None:other.close()
