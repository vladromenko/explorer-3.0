"""Untrained compact upstream Diffusion: save/reload/queues/RNG, never optimize."""
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from lerobot.configs.types import FeatureType,PolicyFeature,NormalizationMode
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from explorer3.paths import DATA
from explorer3.resources import ResourceLease
from explorer3.act_client import PolicyClient


def main():
    root=DATA/'validation/corrective-a/diffusion-inference';root.mkdir(exist_ok=True)
    with ResourceLease('compute'):
        torch.set_num_threads(2);torch.manual_seed(104)
        cfg=DiffusionConfig(device='cpu',push_to_hub=False,
            input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(2,)),
                'observation.images.front':PolicyFeature(type=FeatureType.VISUAL,shape=(3,64,64))},
            output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(2,))},
            normalization_mapping={key:NormalizationMode.IDENTITY for key in ('VISUAL','STATE','ACTION')},
            pretrained_backbone_weights=None,down_dims=(32,64,128),horizon=16,n_action_steps=8,
            num_inference_steps=4,diffusion_step_embed_dim=32)
        checkpoint=root/'untrained_fixture';policy=DiffusionPolicy(cfg).eval()
        pre,post=make_pre_post_processors(cfg)
        policy.save_pretrained(checkpoint);pre.save_pretrained(checkpoint);post.save_pretrained(checkpoint)
        del policy,pre,post;gc.collect()
        client=PolicyClient(checkpoint,root/'inference.log')
        try:
            observation={'state':np.zeros(2,np.float32),'rgb':np.zeros((64,64,3),np.uint8)}
            profile={'camera':'front','task':'Move towards the target and stop.'}
            start=time.monotonic();first=client.action(observation,profile);cold=time.monotonic()-start
            saved=client.request('snapshot')
            expected=[client.action(observation,profile) for _ in range(10)]
            client.request('restore',snapshot=saved)
            actual=[client.action(observation,profile) for _ in range(10)]
            np.testing.assert_array_equal(actual,expected)
            assert np.isfinite(actual).all() and np.isfinite(first).all()
            result={'policy':client.ready,'cold_action_seconds':cold,'replayed_actions':10,
                'snapshot_restore_exact':True,'new_process_reload':True,'training_started':False,
                'random_weights':True,'compact_fixture':True,'skill_verified':False,
                'limitations':['Reduced architecture only; full recipe performance not measured','No optimizer, backward or learning performed']}
            (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
        finally:client.close()

if __name__=='__main__':main()
