"""New-process policy reload and inference; this module cannot send motor commands."""
import json
import sys
import time
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class,make_pre_post_processors


def check(checkpoint: Path, dataset: Path, device='cuda'):
    cfg=PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device=device
    policy=get_policy_class(cfg.type).from_pretrained(checkpoint,config=cfg)
    policy.eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor':{'device':device}},
        postprocessor_overrides={'device_processor':{'device':'cpu'}})
    ds=LeRobotDataset('local/smoke',root=dataset,video_backend='pyav')
    frame=ds[0]
    batch={k:v.unsqueeze(0) for k,v in frame.items() if k.startswith('observation.')}
    batch['task']=[frame['task']]
    policy.reset()
    start=time.monotonic()
    with torch.inference_mode():
        action=post(policy.select_action(pre(batch)))
    assert torch.isfinite(action).all()
    assert action.shape[-1]==ds.features['action']['shape'][0]
    result={'reload':'PASS','inference':'PASS','action_shape':list(action.shape),'latency_seconds':time.monotonic()-start,
            'motor_commands_sent':False,'physical_skill_verified':False}
    (checkpoint.parent/'explorer3-reload.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    return result

if __name__=='__main__':
    check(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else 'cuda')
