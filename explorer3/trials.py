"""Explicit trial plans and action providers, separate from device ownership."""
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from . import catalog, local_work
from .paths import DATA
from .profiles import profile
from .datasets import fingerprint, load_plan


def prepare(kind, ident):
    if kind not in {'replay','policy'}:
        raise ValueError('Unknown trial kind')
    rows=catalog.rows('episodes' if kind=='replay' else 'jobs')
    row=next((r for r in rows if r['id']==ident),None)
    if row is None:raise ValueError('Выберите запись или checkpoint')
    cfg=profile(row['profile'])
    plan={'id':uuid.uuid4().hex,'kind':kind,'profile':row['profile'],'source_id':ident,
          'fps':cfg['fps'],'channels':cfg['channels'],'units':cfg['units'],
          'max_duration_s':60.,'physical_acceptance':False,'motor_commands_sent':False,
          'state':'prepared_requires_physical_block','created':time.time()}
    if kind=='replay':
        from .validation import validate
        validate(ident)
        root=Path(row['dataset'])
        records=[json.loads(line) for line in (root/'research.jsonl').read_text().splitlines()]
        if len({r['command_source'] for r in records})!=1:
            raise ValueError('Replay cannot cross command-source segment boundaries')
        plan.update(dataset=str(root),dataset_sha256=fingerprint(root),frames=len(records),
                    initial_state=records[0]['observation.state'],initial_target=records[0]['action'],
                    max_duration_s=len(records)/cfg['fps'],
                    initial_pose_requirement=('Measured pose must match first recorded target before replay; no automatic move to it'
                        if row['profile'].endswith('pick_place') else 'Operator must restore the starting area and scene; previous velocity commands are not a measured starting pose'))
    else:
        if row['status']!='completed' or not row['checkpoint']:
            raise ValueError('Нужен завершённый trainer с checkpoint')
        job=json.loads(row['config'])
        if job.get('kind','train')!='train':raise ValueError('Выберите именно обучение')
        root=Path(row['checkpoint']).resolve()
        if not root.is_relative_to(DATA/'checkpoints'):raise ValueError('Checkpoint outside Explorer storage')
        weights=root/'model.safetensors'
        train_info=json.loads((DATA/'jobs'/ident/'train/meta/info.json').read_text())
        if train_info['features']['action']['names']!=cfg['channels']:
            raise ValueError('Checkpoint channel order differs from selected task')
        plan.update(checkpoint=str(root),checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                    smoke_checkpoint=bool(job.get('smoke',True)),max_duration_s=60.,
                    limitation='Smoke checkpoint is not a learned physical skill')
    directory=DATA/'trials';directory.mkdir(exist_ok=True)
    (directory/(plan['id']+'.json')).write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    return plan


def load(ident):
    if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid trial ID')
    return json.loads((DATA/'trials'/(ident+'.json')).read_text())


def authorize(ident, hours=3.):
    """Operator's explicit permission for one prepared plan, limited in time."""
    plan=load(ident)
    path=DATA/'configs'/('trial-'+plan['id']+'-authorized.json')
    path.write_text(json.dumps({'trial_id':plan['id'],'profile':plan['profile'],'authorized_at':time.time(),
                                'expires_at':time.time()+hours*3600,'by':'operator in Explorer UI'}))
    return path


def require_physical(plan):
    local_work.require_hardware_mode(profile(plan['profile'])['robot'])
    if plan['profile'] not in {'so101_pick_place','tank_dock','rosmaster_dock','rosmaster_pick_place'}:
        raise PermissionError('Исполнитель этого профиля ещё не принят; план сохранён без движения')
    if plan['profile']=='tank_dock':
        acceptance=DATA/'validation/tank-manual-control-pass.json'
        if (DATA/'validation/tank-motion-incident.json').exists() or not acceptance.exists():
            raise PermissionError('Сначала принять ручное управление и STOP Tank')
        if not json.loads(acceptance.read_text()).get('operator_confirmed_stop'):
            raise PermissionError('STOP Tank ещё не подтверждён оператором')
    approval=DATA/'configs'/('trial-'+plan['id']+'-authorized.json')
    if not approval.exists():
        raise PermissionError('Нужен отдельный физический блок для этого плана испытания')
    value=json.loads(approval.read_text())
    if value.get('trial_id')!=plan['id'] or value.get('expires_at',0)<=time.time():
        raise PermissionError('Разрешение на испытание отсутствует или истекло')


class ReplayActions:
    def __init__(self,plan):
        if fingerprint(plan['dataset'])!=plan['dataset_sha256']:raise ValueError('Replay dataset changed')
        self.rows=[json.loads(line) for line in (Path(plan['dataset'])/'research.jsonl').read_text().splitlines()]
        self.index=0
        self.check_pose=plan.get('profile','so101_pick_place').endswith('pick_place')

    def next(self,observation):
        if self.index>=len(self.rows):raise StopIteration
        action=self.rows[self.index]['action']
        if self.check_pose and self.index==0 and any(abs(a-b)>(3. if i==5 else 2.) for i,(a,b) in enumerate(zip(action,observation['observation.state']))):
            raise ValueError('Начальная поза отличается от записи. Автоматический переход не выполняется')
        self.index+=1
        return list(action)

    def reset(self):
        self.index=0


class PolicyActions:
    def __init__(self,plan):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class,make_pre_post_processors
        root=Path(plan['checkpoint'])
        if hashlib.sha256((root/'model.safetensors').read_bytes()).hexdigest()!=plan['checkpoint_sha256']:
            raise ValueError('Policy checkpoint changed')
        cfg=PreTrainedConfig.from_pretrained(root);cfg.device='cuda'
        self.policy=get_policy_class(cfg.type).from_pretrained(root,config=cfg)
        self.policy.eval()
        self.pre,self.post=make_pre_post_processors(cfg,pretrained_path=str(root),
            preprocessor_overrides={'device_processor':{'device':'cuda'}},
            postprocessor_overrides={'device_processor':{'device':'cpu'}})
        self.task=profile(plan['profile'])['task']
        self.decision_info={}
        self.policy.reset()

    def needs_inference(self):
        queue=getattr(self.policy,'_action_queue',getattr(self.policy,'_queues',{}).get('action'))
        return queue is None or len(queue)==0

    def next(self,observation):
        import torch
        import numpy as np
        batch={'observation.state':torch.tensor([observation['observation.state']],dtype=torch.float32),'task':[self.task]}
        for name,image in observation['images'].items():
            rgb=np.array(image['image'],copy=True)
            batch['observation.images.'+name]=torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0)/255.
        replanned=self.needs_inference()
        start=time.monotonic_ns()
        with torch.inference_mode():
            action=self.post(self.policy.select_action(self.pre(batch))).cpu().numpy()[0].tolist()
        end=time.monotonic_ns()
        if replanned:
            self.decision_info={'policy_observation_timestamp_ns':observation.get('observation_timestamp_ns'),
                                'policy_generation_start_ns':start,'policy_generation_end_ns':end}
        self.decision_info['policy_replanned']=replanned
        if not all(math.isfinite(a) for a in action):raise ValueError('Nonfinite policy action')
        return action

    def reset(self):
        self.policy.reset()
        self.decision_info={}
