"""Public-base SmolVLA forward/backward/save/new-process reload, software only."""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from explorer3.paths import DATA,offline_environment
from explorer3.resources import ResourceLease
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

with ResourceLease('compute'):
    base=next((DATA/'cache/huggingface/hub/models--lerobot--smolvla_base/snapshots').iterdir())
    vlm=next((DATA/'cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots').iterdir())
    act=json.loads((DATA/'validation/act-pass.json').read_text())
    root=Path(act['run'])/'synthetic_dataset'
    ds=LeRobotDataset('local/synthetic_software_only',root=root,video_backend='pyav',delta_timestamps={'action':[i/30 for i in range(50)]})
    cfg=SmolVLAConfig.from_pretrained(base)
    cfg.input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(6,)),
                        'observation.images.wrist':PolicyFeature(type=FeatureType.VISUAL,shape=(3,64,64))}
    cfg.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))}
    cfg.vlm_model_name=str(vlm)
    cfg.device='cuda'
    cfg.push_to_hub=False
    cfg.load_vlm_weights=True
    policy=SmolVLAPolicy.from_pretrained(base,config=cfg)
    pre,post=make_pre_post_processors(cfg,dataset_stats=ds.meta.stats)
    frame=ds[0]
    batch={k:v.unsqueeze(0) for k,v in frame.items() if k.startswith('observation.') or k in {'action','action_is_pad'}}
    batch['task']=['Pick up the cube, place it in the box, and withdraw the arm.']
    optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=cfg.optimizer_lr)
    policy.train()
    loss,_=policy(pre(batch))
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg.optimizer_grad_clip_norm)
    optimizer.step()
    run=DATA/'validation'/('smolvla-'+str(int(time.time())))
    checkpoint=run/'pretrained_model'
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    loss_value=float(loss.detach().cpu())
    del policy,optimizer,pre,post,batch,loss
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-m','explorer3.reload_check',str(checkpoint),str(root)],env=offline_environment(),check=True)
    (DATA/'validation/smolvla-pass.json').write_text(json.dumps({'run':str(run),'synthetic':True,'loss':loss_value,
        'forward_backward_reload':'PASS','physical_skill_verified':False,'batch':1}))
    print('SmolVLA SOFTWARE PASS',flush=True)
