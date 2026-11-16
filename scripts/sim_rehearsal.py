"""Rehearsal of the whole learning pipeline in MuJoCo before the real demonstrations exist.

For each arm skill (SO-101 hand-over pick, ROSMASTER floor pick) and each policy (SmolVLA, ACT):
  1. expert demonstrations in MuJoCo, written in exactly the real dataset format (sim_of profiles);
  2. training through Explorer's own job queue (same trainer as for real data);
  3. closed-loop evaluation on unseen MuJoCo seeds through explorer3.policy_runtime — the runtime
     that real trials and the swarm use;
  4. the trained checkpoint on REAL robot inputs, no motors: recorded SO-101 frames, and live
     ROSMASTER camera + servo readings in shadow mode (actions computed, never sent).
Writes DATA/simlab/rehearsal/<time>/report.json. Run with the main env; MuJoCo runs in .venv-sim.

  .venv/bin/python scripts/sim_rehearsal.py            # full (overnight)
  .venv/bin/python scripts/sim_rehearsal.py --smoke    # pipeline check in minutes
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explorer3 import catalog  # noqa: E402
from explorer3.paths import DATA, offline_environment  # noqa: E402
from explorer3.profiles import profile  # noqa: E402

# the MuJoCo tasks that mirror recording blocks 4 and 3 (explorer3/sim/aligned.py)
SKILLS = ['so101_handover_sim', 'rosmaster_floor_pick_sim']
POLICIES = ['smolvla', 'act']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def current_model(name):
    """Physics version of the newest demonstration (old proxy-geometry versions are not pooled)."""
    from explorer3.datasets import simulation_model_hash
    eps = sorted((e for e in catalog.rows('episodes') if e['profile'] == name), key=lambda e: e['created'])
    return simulation_model_hash(eps[-1]) if eps else None


def successes(name, model=None):
    from explorer3.datasets import simulation_model_hash
    model = model or current_model(name)
    return [e for e in catalog.rows('episodes') if e['profile'] == name and e['outcome'] == 'success' and not e.get('excluded')
            and simulation_model_hash(e) == model]


def collect(name, need, workers=2):          # 3 MuJoCo renderers + a model hit the 16 GB OOM limit
    # Read today's physics, not the most recently recorded (possibly obsolete) episode.
    probe = subprocess.run([str(ROOT / '.venv-sim/bin/python'), '-c',
                            'import sys; from explorer3.sim.factory import make; '
                            'e=make(sys.argv[1]); print(e.manifest["xml_sha256"]); e.close()', name],
                           cwd=ROOT, env=offline_environment(), capture_output=True, text=True,
                           check=True, timeout=60)
    model = probe.stdout.strip()
    if len(model) != 64 or any(c not in '0123456789abcdef' for c in model):
        raise RuntimeError('Не удалось определить текущую версию сцены MuJoCo')
    have = len(successes(name, model))
    if have >= need:
        log(name, f'показов уже {have}')
        return have
    env = {**offline_environment(), 'MUJOCO_GL': 'egl'}
    for round_ in range(4):                       # the scripted expert fails some scenes: try until enough
        missing = need - have
        if missing <= 0:
            break
        tries = math.ceil(missing * (1.3 if round_ == 0 else 2.2))
        per = math.ceil(tries / workers)
        start = 2000 + (int(time.time()) + round_ * 7919) % 100000          # fresh seeds every run
        procs = [subprocess.Popen([str(ROOT / '.venv-sim/bin/python'), '-m', 'explorer3.sim.collect', '--start', str(start + i * per),
                                   '--count', str(per), '--profile', name], cwd=ROOT, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(workers)]
        log(name, f'не хватает {missing} показов: {tries} попыток эксперта в MuJoCo ({workers} процесса)')
        for p in procs:
            p.wait()
        have = len(successes(name, model))
        log(name, f'показов теперь {have}')
    if have < need:
        raise RuntimeError(f'{name}: получено {have}/{need} показов текущей сцены; обучение не запускается')
    return have


def wait_job(ident, timeout_h=8):
    end = time.time() + timeout_h * 3600
    last = None
    while time.time() < end:
        row = next(j for j in catalog.rows('jobs') if j['id'] == ident)
        if row['status'] != last:
            log('  задание', ident[:8], row['status'], row.get('error') or '')
            last = row['status']
        if row['status'] in ('completed', 'failed', 'cancelled'):
            return row
        time.sleep(20)
    raise TimeoutError('job ' + ident)


def real_inputs_check(checkpoint, real_profile, instruction=None, shadow_steps=40):
    """The trained policy through the shared runtime on real robot inputs. No motor command is sent."""
    import numpy as np
    from explorer3.policy_runtime import PolicyCore
    cfg = profile(real_profile)
    core = PolicyCore(checkpoint, 'cuda', instruction=instruction)
    out = {'profile': real_profile, 'policy': core.type, 'source': None, 'steps': 0, 'latency_s_p50_max': None,
           'finite': True, 'max_jump_deg': None, 'within_limits': None, 'motor_commands_sent': 0}
    lat, actions, states = [], [], []
    if real_profile == 'so101_pick_place':
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        recorded = [e for e in catalog.rows('episodes') if e['profile'] == 'so101_pick_place' and e['outcome'] == 'success']
        if not recorded:
            out['source'] = 'нет записанных показов SO-101'
            return out
        root = Path(recorded[0]['dataset'])
        ds = LeRobotDataset('local/' + root.name, root=root, video_backend='pyav')
        out['source'] = f'записанный показ SO-101 {root.name[:8]} ({len(ds)} кадров)'
        key = 'observation.images.' + cfg['camera']
        for i in range(0, min(len(ds), shadow_steps * 3), 3):
            f = ds[i]
            rgb = (f[key].permute(1, 2, 0).numpy() * 255).astype('uint8')
            state = f['observation.state'].tolist()
            t = time.monotonic()
            a = core.act(state, {cfg['camera']: rgb}, cfg['task'])
            lat.append(time.monotonic() - t)
            actions.append(a)
            states.append(state)
    elif real_profile == 'rosmaster_pick_place':
        import cv2
        from explorer3 import rosmaster_control
        try:
            client = rosmaster_control._connection(4)
        except Exception as exc:
            out['source'] = f'ROSMASTER недоступен: {exc}'
            return out
        out['source'] = 'живой ROSMASTER, теневой режим (действия считаются, не отправляются)'
        try:
            for _ in range(shadow_steps):
                started = time.monotonic()
                try:                                     # a slow Wi-Fi reply skips one step, it is not a failure
                    arm = (client.get('/health').json().get('observation') or {}).get('arm_measured')
                    frame = client.get('/frame.jpg')
                except Exception:
                    out['network_skips'] = out.get('network_skips', 0) + 1
                    continue
                if not arm or frame.status_code != 200:
                    continue
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                h, w = cfg['image_shape'][:2] if cfg.get('image_shape') else rgb.shape[:2]
                if rgb.shape[:2] != (h, w):
                    rgb = cv2.resize(rgb, (w, h))
                t = time.monotonic()
                a = core.act([float(v) for v in arm], {cfg['camera']: rgb}, cfg['task'])
                lat.append(time.monotonic() - t)
                actions.append(a)
                states.append([float(v) for v in arm])
                time.sleep(max(0., 1 / cfg['fps'] - (time.monotonic() - started)))
        finally:
            client.close()
    if actions:
        A, S = np.array(actions), np.array(states)
        out.update(steps=len(actions), latency_s_p50_max=[round(float(np.median(lat)), 3), round(float(max(lat)), 3)],
                   finite=bool(np.isfinite(A).all()), max_jump_deg=round(float(np.abs(A[:, :5] - S[:, :5]).max()), 1))
        if real_profile == 'rosmaster_pick_place':
            from explorer3.sim.vendor_rosmaster.arm_model import SERVO_LIMITS
            out['within_limits'] = bool(all(lo - 1 <= v <= hi + 1 for a in A for v, (lo, hi) in zip(a, SERVO_LIMITS)))
        else:
            out['within_limits'] = bool((np.abs(A[:, :5]) <= 180).all() and (A[:, 5] >= -1).all() and (A[:, 5] <= 101).all())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--skills', default=','.join(SKILLS))
    ap.add_argument('--policies', default=','.join(POLICIES))
    ap.add_argument('--collect-only', action='store_true')
    ap.add_argument('--demonstrations', type=int, help='Required successes per skill (default: 50, smoke: 6)')
    ap.add_argument('--workers', type=int, choices=(1, 2), default=1,
                    help='Concurrent capture processes; one by default for the 6 GiB scope')
    a = ap.parse_args()
    out = DATA / 'simlab' / 'rehearsal' / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    from explorer3.jobs import Jobs
    jobs = Jobs(None)
    report = {'started': time.time(), 'smoke': a.smoke, 'results': []}
    need = a.demonstrations if a.demonstrations is not None else 6 if a.smoke else 50
    if need < (3 if a.collect_only or a.smoke else 50):
        ap.error('At least 3 demonstrations for collection/smoke; at least 50 for training')
    for name in a.skills.split(','):
        collect(name, need, workers=a.workers)
    if a.collect_only:
        return
    for name in a.skills.split(','):
        real = profile(name).get('real_profile')
        for policy in a.policies.split(','):
            entry = {'skill': name, 'real_profile': real, 'policy': policy}
            report['results'].append(entry)
            try:
                log(name, policy, 'обучение через очередь Explorer', '(smoke)' if a.smoke else '')
                model = current_model(name)
                entry['physics_model'] = model
                ident = jobs.enqueue(name, policy, smoke=a.smoke, sim_model_hash=model)
                train = wait_job(ident, 2 if a.smoke else 10)
                entry['train'] = {'job': ident, 'status': train['status'], 'checkpoint': train['checkpoint'], 'error': train.get('error')}
                if train['status'] != 'completed':
                    continue
                log(name, policy, 'проверка в MuJoCo на новых сценах')
                ev = jobs.sim_evaluate(ident)
                evrow = wait_job(ev, 2)
                result = DATA / 'jobs' / ev / 'result.json'
                sim = json.loads(result.read_text()) if result.exists() else {}
                entry['mujoco'] = {'job': ev, 'status': evrow['status'], 'episodes': len(sim.get('trials', [])),
                                   'success_fraction': sim.get('success_fraction'),
                                   'latency_s_p50_p95_max': sim.get('sensor_to_action_seconds_p50_p95_max')}
                log(name, policy, 'успех в MuJoCo:', entry['mujoco']['success_fraction'])
                log(name, policy, 'реальные входы робота (без моторов)')
                entry['real_inputs'] = real_inputs_check(train['checkpoint'], real)
                log(name, policy, json.dumps(entry['real_inputs'], ensure_ascii=False))
                log(name, policy, '«Допуск к роботу»')
                from explorer3 import deploy_gate
                gate = deploy_gate.run(ident)
                entry['deploy_gate'] = {'verdict': gate['verdict'], 'checks': {k: {'status': c['status'], 'message': c['message']}
                                                                              for k, c in gate['checks'].items()}}
                log(name, policy, 'допуск:', gate['verdict'], '·', gate['meaning'][:300])
            except Exception as exc:
                entry['error'] = f'{type(exc).__name__}: {exc}'
                log(name, policy, 'ОШИБКА', entry['error'])
            (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    report['finished'] = time.time()
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log('отчёт:', out / 'report.json')


if __name__ == '__main__':
    main()
