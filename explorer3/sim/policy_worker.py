"""Official LeRobot ACT / SmolVLA / Diffusion inference in the existing environment. No hardware imports."""
import argparse
import base64
import json
import hashlib
import sys
from pathlib import Path
import cv2
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class,make_pre_post_processors


def main(checkpoint,device):
    torch.set_num_threads(2)
    cfg=PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type not in {'act','smolvla','diffusion'}:
        raise ValueError('Unsupported LeRobot policy')
    if cfg.type=='smolvla' and getattr(cfg,'rtc_config',None) is not None:
        raise ValueError('RTC state is not yet supported by this bounded evaluator')
    cfg.device=device
    policy=get_policy_class(cfg.type).from_pretrained(checkpoint,config=cfg).eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor':{'device':device}},
        postprocessor_overrides={'device_processor':{'device':'cpu'}})
    policy.reset()
    import numpy as np
    fingerprint=hashlib.sha256(b''.join(p.name.encode()+hashlib.sha256(p.read_bytes()).digest()
        for p in sorted(checkpoint.iterdir()) if p.is_file())).hexdigest()
    def tensor_state(value):
        return None if value is None else {'values':value.detach().cpu().tolist(),'dtype':str(value.dtype).split('.')[-1]}
    def restore_tensor(value):
        return None if value is None else torch.tensor(value['values'],dtype=getattr(torch,value['dtype']),device=device)
    print('EXPLORER_JSON '+json.dumps({'ready':True,'policy_id':fingerprint,
        'policy_type':cfg.type,'features':f'ACT {cfg.vision_backbone} layer4 spatial mean, frozen' if cfg.type=='act' else None,
        'chunk_size':getattr(cfg,'chunk_size',getattr(cfg,'horizon',None))}),flush=True)
    for line in sys.stdin:
        req=json.loads(line)
        if req['command']=='reset':
            policy.reset();result={'reset':True}
        elif req['command']=='snapshot':
            ensemble=getattr(policy,'temporal_ensembler',None)
            result={'policy_id':fingerprint,'queue':[tensor_state(x) for x in getattr(policy,'_action_queue',[])],
                'ensemble':None if ensemble is None else {name:tensor_state(getattr(ensemble,name))
                    for name in ('ensembled_actions','ensembled_actions_count')},
                'torch_rng':torch.get_rng_state().tolist(),
                'queues':{key:[tensor_state(x) for x in value] for key,value in getattr(policy,'_queues',{}).items()}}
            if device.startswith('cuda'):result['cuda_rng']=[x.tolist() for x in torch.cuda.get_rng_state_all()]
        elif req['command']=='restore':
            saved=req['snapshot']
            if saved['policy_id']!=fingerprint:raise ValueError('Policy snapshot belongs to another checkpoint')
            policy.reset()
            if saved['ensemble'] is not None:
                for name,value in saved['ensemble'].items():setattr(policy.temporal_ensembler,name,restore_tensor(value))
            elif cfg.type=='act':policy._action_queue.extend(restore_tensor(x) for x in saved['queue'])
            for key,values in saved.get('queues',{}).items():policy._queues[key].extend(restore_tensor(x) for x in values)
            torch.set_rng_state(torch.tensor(saved['torch_rng'],dtype=torch.uint8))
            if 'cuda_rng' in saved:torch.cuda.set_rng_state_all([torch.tensor(x,dtype=torch.uint8) for x in saved['cuda_rng']])
            result={'restored':True}
        elif req['command'] in {'action','propose','embed'}:
            bgr=cv2.imdecode(np.frombuffer(base64.b64decode(req['rgb']),dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError('Invalid camera')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            batch={'observation.state':torch.tensor(req['state'],dtype=torch.float32).unsqueeze(0),
                   req['camera_key']:torch.from_numpy(rgb.copy()).permute(2,0,1).float().unsqueeze(0)/255,
                   'task':[req['task']]}
            with torch.inference_mode():
                prepared=pre(batch)
                if req['command']=='action':
                    action=post(policy.select_action(prepared))
                    result={'action':action.reshape(-1).tolist()}
                elif req['command']=='embed':
                    if cfg.type!='act':raise ValueError('Support embeddings calibrated only for ACT')
                    embeddings=[policy.model.backbone(prepared[key])['feature_map'].mean(dim=(-2,-1)).cpu() for key in cfg.image_features]
                    result={'policy_id':fingerprint,'features':torch.stack(embeddings,dim=1)[0].tolist(),'queue_unchanged':True}
                else:
                    if cfg.type!='act':raise ValueError('Support proposal calibrated only for ACT')
                    features=[]
                    def collect_features(module,inputs,output):
                        features.append(output['feature_map'].mean(dim=(-2,-1)).detach().cpu())
                    hook=policy.model.backbone.register_forward_hook(collect_features)
                    try:chunk=post(policy.predict_action_chunk(prepared))
                    finally:hook.remove()
                    result={'policy_id':fingerprint,'features':torch.stack(features,dim=1)[0].tolist(),
                        'chunk':chunk[0].cpu().tolist(),'proposal_kind':'policy_prediction',
                        'queue_unchanged':True}
        else:
            raise ValueError('Unknown policy command')
        print('EXPLORER_JSON '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();main(args.checkpoint,args.device)
