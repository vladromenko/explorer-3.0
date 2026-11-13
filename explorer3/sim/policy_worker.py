"""Official ACT inference in the existing environment. No hardware imports."""
import argparse
import base64
import json
import sys
from pathlib import Path
import cv2
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class,make_pre_post_processors


def main(checkpoint,device):
    cfg=PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type!='act':
        raise ValueError('ACT checkpoint required')
    cfg.device=device
    policy=get_policy_class('act').from_pretrained(checkpoint,config=cfg).eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor':{'device':device}},
        postprocessor_overrides={'device_processor':{'device':'cpu'}})
    policy.reset()
    import numpy as np
    print('EXPLORER_JSON '+json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        req=json.loads(line)
        if req['command']=='reset':
            policy.reset();result={'reset':True}
        elif req['command']=='action':
            bgr=cv2.imdecode(np.frombuffer(base64.b64decode(req['rgb']),dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError('Invalid camera')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            batch={'observation.state':torch.tensor(req['state'],dtype=torch.float32).unsqueeze(0),
                   req['camera_key']:torch.from_numpy(rgb.copy()).permute(2,0,1).float().unsqueeze(0)/255,
                   'task':[req['task']]}
            with torch.inference_mode():
                action=post(policy.select_action(pre(batch)))
            result={'action':action.reshape(-1).tolist()}
        else:
            raise ValueError('Unknown policy command')
        print('EXPLORER_JSON '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();main(args.checkpoint,args.device)
