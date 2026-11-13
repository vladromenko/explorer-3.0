"""ROSMASTER data -> plan -> checkpoint -> trial readiness, WITHOUT training and WITHOUT the robot.

Run in an isolated data dir that already holds ROSMASTER episodes and checkpoints, e.g.
  EXPLORER3_DATA=~/.local/share/explorer3/pipeline-check/<stamp> .venv/bin/python scripts/check_rosmaster_flow.py
Nothing is sent to a robot: the policy is loaded and asked for one action on a RECORDED frame.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from explorer3.paths import DATA
from explorer3 import catalog, trials
from explorer3.validation import validate
from explorer3.datasets import freeze

assert 'pipeline-check' in str(DATA) or 'fixture' in str(DATA), 'isolated data dir only'
report = {'data': str(DATA), 'training_started': False, 'motor_commands_sent': False, 'profiles': {}}
for name in ('rosmaster_dock', 'rosmaster_pick_place'):
    out = report['profiles'][name] = {}
    episodes = [e for e in catalog.rows('episodes') if e['profile'] == name]
    out['readback'] = [validate(e['id'])['readback'] for e in episodes]
    plan_id, plan = freeze(name, smoke=True)
    again, _ = freeze(name, smoke=True)
    out['dataset_plan'] = {'id': plan_id, 'immutable_same_id_on_refreeze': again == plan_id,
                           'train_episodes': len(plan['splits']['train']), 'limitation': plan['limitation']}
    job = next(j for j in catalog.rows('jobs') if j['profile'] == name and j['status'] == 'completed' and j['checkpoint'])
    policy_plan = trials.prepare('policy', job['id'])
    out['policy_trial_plan'] = {k: policy_plan[k] for k in ('id', 'state', 'physical_acceptance', 'smoke_checkpoint', 'limitation')}
    try:
        trials.require_physical(policy_plan); out['physical_start_gate'] = 'OPEN (unexpected)'
    except PermissionError as exc:
        out['physical_start_gate'] = 'closed: ' + str(exc)
    exception = DATA / 'configs' / 'rosmaster-hardware-enabled.json'; exception.parent.mkdir(exist_ok=True)
    exception.write_text('{"enabled": true}')
    try:
        try:
            trials.require_physical(policy_plan); out['gate_with_exception_without_authorization'] = 'OPEN (unexpected)'
        except PermissionError as exc:
            out['gate_with_exception_without_authorization'] = 'closed: ' + str(exc)
        approval = trials.authorize(policy_plan['id'])
        trials.require_physical(policy_plan); out['gate_with_exception_and_authorization'] = 'open (plan only; nothing executed)'
        approval.unlink()
    finally:
        exception.unlink()
    provider = trials.PolicyActions(policy_plan)
    root = Path(episodes[0]['dataset'])
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    frame = LeRobotDataset('local/check', root=root, video_backend='pyav')[0]
    image = (frame['observation.images.front'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    action = provider.next({'observation.state': frame['observation.state'].tolist(), 'images': {'front': {'image': image}}})
    provider.reset()
    out['policy_inference_on_recorded_frame'] = {'action': [round(float(v), 4) for v in action], 'finite': all(math.isfinite(v) for v in action),
                                                 'dims_match_profile': len(action) == len(policy_plan['channels'])}
    replay = trials.prepare('replay', episodes[0]['id'])
    out['replay_plan'] = {'id': replay['id'], 'frames': replay['frames'], 'state': replay['state']}
(DATA / 'rosmaster-flow-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
