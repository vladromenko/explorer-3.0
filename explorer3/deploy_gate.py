"""«Допуск к роботу»: what has to be true about a trained checkpoint before it may drive a real arm.

A policy trained in MuJoCo or on real demonstrations is admitted to a robot only when ALL of:
  1. contract   — its inputs/outputs are exactly the robot's: joint channels and order, state size,
                  camera name and image size, fps; SmolVLA has its language instruction;
  2. latency    — on this Orin, a full replan through the robot runtime (explorer3.policy_runtime,
                  the same code real trials use) fits into the time its action chunk lasts;
  3. real data  — on REAL demonstrations that were not in its training set, run through the same
                  runtime: the predicted joint targets are close to what the person did (MAE), and
                  the first target is close to where the arm actually is (no jump at the start);
  4. MuJoCo     — on the MuJoCo task that mirrors the recording block (explorer3/sim/aligned.py),
                  closed loop on unseen seeds, success at least the threshold.
Thresholds: configs/deploy-gate.json. The result is bound to the checkpoint bundle hash; changed
weights need a new check. What it does NOT prove: success on the real robot — that is measured by
the first real trials (short, in the limited trial mode), never assumed.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import catalog
from .paths import DATA, ROOT
from .profiles import profile

GATES = DATA / 'gates'


def thresholds():
    return json.loads((ROOT / 'configs/deploy-gate.json').read_text())


def real_profile_of(name):
    cfg = profile(name)
    return cfg.get('real_profile', name) if cfg.get('domain') == 'sim' else name


def training_ids(job):
    cfg = json.loads(job['config'])
    if cfg.get('plan_id'):
        from .datasets import load_plan
        splits = load_plan(cfg['plan_id'])['splits']
    else:
        path = DATA / 'jobs' / job['id'] / 'split.json'
        splits = json.loads(path.read_text()) if path.exists() else {'train': []}
    return {e['id'] for e in splits.get('train', [])}


def holdout_episodes(job, real):
    """Real demonstrations of the robot's task that the policy never trained on."""
    from .datasets import in_task_block, series_text
    seen = training_ids(job)
    texts = series_text()
    return [e for e in catalog.rows('episodes') if e['profile'] == real and e['outcome'] == 'success'
            and not e.get('excluded') and e['id'] not in seen and e.get('category', 'demonstration') == 'demonstration'
            and in_task_block(e, texts)]


def check_contract(core, real, job_id):
    cfg = profile(real)
    problems = []
    feats = core.cfg.input_features
    state = feats.get('observation.state')
    want_state = len(cfg.get('state_channels') or cfg['channels'])
    if state is None or state.shape[0] != want_state:
        problems.append(f'размер состояния {None if state is None else state.shape[0]}, у робота {want_state}')
    action = core.cfg.output_features.get('action')
    if action is None or action.shape[0] != len(cfg['channels']):
        problems.append(f'размер действия {None if action is None else action.shape[0]}, у робота {len(cfg["channels"])}')
    key = 'observation.images.' + cfg['camera']
    if key not in feats:
        problems.append(f'нет камеры {cfg["camera"]} (у модели: {", ".join(k.split(".")[-1] for k in core.image_keys)})')
    info = DATA / 'jobs' / job_id / 'train/meta/info.json'
    if info.exists():
        names = json.loads(info.read_text())['features']['action']['names']
        if names != cfg['channels']:
            problems.append('порядок суставов отличается от робота')
        fps = json.loads(info.read_text()).get('fps')
        if fps and int(fps) != int(cfg['fps']):
            problems.append(f'частота {fps} Гц, у робота {cfg["fps"]} Гц')
    return {'status': 'fail' if problems else 'pass',
            'message': '; '.join(problems) or f'входы и выходы совпадают с {real}: {len(cfg["channels"])} суставов, камера {cfg["camera"]}, {cfg["fps"]} Гц'}


def replay_real(core, episodes, real, limits):
    """The policy on real held-out frames through the robot runtime (no motors)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    cfg = profile(real)
    key = 'observation.images.' + cfg['camera']
    errors, jumps, latency = [], [], []
    for e in episodes:
        root = Path(e['dataset'])
        ds = LeRobotDataset('local/' + root.name, root=root, video_backend='pyav')
        for i in np.linspace(0, len(ds) - 1, min(len(ds), limits['frames_per_episode'])).astype(int):
            f = ds[int(i)]
            rgb = (f[key].permute(1, 2, 0).numpy() * 255).astype('uint8')
            state = f['observation.state'].tolist()
            core.reset()                                   # each sample is a fresh replan
            t = time.monotonic()
            a = np.array(core.act(state, {cfg['camera']: rgb}, f.get('task') or cfg['task']))
            latency.append(time.monotonic() - t)
            errors.append(np.abs(a - f['action'].numpy().reshape(-1)[:len(a)]))
            jumps.append(float(np.abs(a[:5] - np.array(state[:5])).max()))
    return np.array(errors), np.array(jumps), np.array(latency)


def mujoco_result(job_id):
    rows = [j for j in catalog.rows('jobs') if j['status'] == 'completed' and json.loads(j['config']).get('kind') == 'sim_evaluate'
            and json.loads(j['config']).get('source_job') == job_id]
    if not rows:
        return None
    row = max(rows, key=lambda j: j['created'])
    path = DATA / 'jobs' / row['id'] / 'result.json'
    return {**json.loads(path.read_text()), 'job': row['id'], 'profile': json.loads(row['config'])['profile']} if path.exists() else None


def run(job_id, out_dir=None):
    from .policy_runtime import PolicyCore
    from .training_drafts import checkpoint_hash
    limits = thresholds()
    job = next(j for j in catalog.rows('jobs') if j['id'] == job_id)
    if job['status'] != 'completed' or not job['checkpoint']:
        raise ValueError('Нужна завершённая тренировка с checkpoint')
    real = real_profile_of(job['profile'])
    cfg = profile(real)
    checks = {}
    core = PolicyCore(job['checkpoint'], 'cuda')
    checks['contract'] = check_contract(core, real, job_id)

    held = holdout_episodes(job, real)
    if len(held) < limits['min_real_holdout_episodes']:
        checks['real_holdout'] = {'status': 'blocked', 'message': f'нужно ≥{limits["min_real_holdout_episodes"]} настоящих показов «{real}», '
                                  f'которых не было в обучении (есть {len(held)}). Без них нельзя сказать, что модель поймёт настоящую камеру и руку.'}
        checks['first_step'] = {'status': 'blocked', 'message': 'проверяется на тех же настоящих показах'}
        checks['latency'] = {'status': 'not_run', 'message': 'измеряется на настоящих кадрах'}
    else:
        errors, jumps, latency = replay_real(core, held, real, limits)
        joints = float(np.median(errors[:, :5].mean(1)))
        grip = float(np.median(errors[:, 5])) if errors.shape[1] > 5 else 0.
        ok = joints <= limits['joint_mae_deg'] and grip <= limits['gripper_mae']
        checks['real_holdout'] = {'status': 'pass' if ok else 'fail', 'episodes': len(held), 'frames': int(len(errors)),
                                  'joint_mae_deg': round(joints, 2), 'gripper_mae': round(grip, 2),
                                  'message': f'на {len(held)} настоящих показах вне обучения: суставы ±{joints:.1f}° '
                                             f'(допуск {limits["joint_mae_deg"]}°), захват ±{grip:.1f} (допуск {limits["gripper_mae"]})'}
        p95 = float(np.percentile(jumps, 95))
        checks['first_step'] = {'status': 'pass' if p95 <= limits['first_step_jump_deg_p95'] else 'fail', 'jump_deg_p95': round(p95, 1),
                                'message': f'первая цель отстоит от текущей позы руки на {p95:.0f}° (95-й процентиль, допуск {limits["first_step_jump_deg_p95"]}°)'}
        steps = getattr(core.cfg, 'n_action_steps', 1) or 1
        budget = steps / cfg['fps'] * limits['latency_budget_fraction']
        lp95 = float(np.percentile(latency, 95))
        checks['latency'] = {'status': 'pass' if lp95 <= budget else 'fail', 'replan_s_p95': round(lp95, 3), 'budget_s': round(budget, 3),
                             'message': f'перепланирование {lp95:.2f} с (95%), запас пачки действий {budget:.2f} с'}

    sim = mujoco_result(job_id)
    if sim is None:
        checks['mujoco'] = {'status': 'not_run', 'message': 'проверка в MuJoCo на задаче, повторяющей блок записи, ещё не запускалась'}
    else:
        n, s = len(sim.get('trials', [])), sim.get('success_fraction') or 0.
        ok = n >= limits['mujoco_min_episodes'] and s >= limits['mujoco_min_success']
        checks['mujoco'] = {'status': 'pass' if ok else 'fail', 'success': s, 'episodes': n, 'job': sim['job'], 'scene': sim['profile'],
                            'message': f'MuJoCo «{sim["profile"]}»: успех {s:.0%} на {n} новых сценах (нужно ≥{limits["mujoco_min_success"]:.0%} на ≥{limits["mujoco_min_episodes"]})'}

    ready = all(c['status'] == 'pass' for c in checks.values())
    report = {'job': job_id, 'checkpoint': job['checkpoint'], 'checkpoint_bundle_sha256': checkpoint_hash(job['checkpoint']),
              'trained_on': job['profile'], 'robot_task': real, 'policy': core.type, 'checked_at': time.time(),
              'verdict': 'ready' if ready else 'blocked', 'checks': checks, 'thresholds': limits,
              'motor_commands_sent': 0,
              'meaning': ('Допущена к первым испытаниям на роботе в ограниченном режиме: входы совпадают, успевает по времени, '
                          'на настоящих показах вне обучения повторяет человека, в MuJoCo справляется. Успех на роботе измеряется испытаниями.')
              if ready else 'Не допущена: ' + '; '.join(f'{k}: {c["message"]}' for k, c in checks.items() if c['status'] != 'pass')}
    GATES.mkdir(parents=True, exist_ok=True)
    (GATES / f'{job_id}.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if out_dir is not None:
        (Path(out_dir) / 'result.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def latest(job_id):
    path = GATES / f'{job_id}.json'
    return json.loads(path.read_text()) if path.exists() else None


def require(plan):
    """Called before a real policy trial: the gate must have passed for exactly these weights."""
    from .training_drafts import checkpoint_hash
    report = latest(plan['source_id'])
    if report is None:
        raise PermissionError('Модель не проверена: запустите «Допуск к роботу» для этой тренировки')
    if report['checkpoint_bundle_sha256'] != checkpoint_hash(plan['checkpoint']):
        raise PermissionError('Веса изменились после «Допуска к роботу»: проверьте заново')
    if report['verdict'] != 'ready':
        raise PermissionError(report['meaning'])
    return report
