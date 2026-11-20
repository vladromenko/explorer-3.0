"""E3 diagnostic (review item): are the training data themselves correct?

Open-loop replay of recorded actions on the same seed in MuJoCo. If the recorded action column is the
command that produced the success, replay must succeed too. Also checks state/action semantics: action[t]
is followed by state[t+1] (lag), joint ranges and units, gripper open/close values, and first-frame
pixels against a fresh render of the same seed (catches RGB/BGR or camera swaps).

  .venv/bin/python scripts/e3_replay_check.py --dataset rosmaster_floor_pick_knowledge_sim --episodes 20
  (it calls .venv-sim for the MuJoCo part; CPU only, quiet)
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3.paths import DATA  # noqa: E402


def extract(folder):
    import av
    import pandas as pd
    df = pd.read_parquet(folder / 'data/chunk-000/file-000.parquet')
    prov = json.loads((folder / 'provenance.json').read_text())
    video = next((folder / 'videos/observation.images.front').rglob('*.mp4'))
    with av.open(str(video)) as c:
        frame = next(c.decode(video=0)).to_ndarray(format='rgb24')
    return {'episode': folder.name, 'seed': prov['hardware']['seed'], 'profile': prov['hardware']['profile'],
            'task': df['task_index'].iloc[0].item() if 'task_index' in df else None,
            'action': [list(map(float, a)) for a in df['action']],
            'state': [list(map(float, s)) for s in df['observation.state']],
            'frame_mean_rgb': frame.reshape(-1, 3).mean(0).round(2).tolist(),
            'frame_small': frame[::16, ::16].tolist()}


def replay(item):
    import numpy as np
    from explorer3.sim.factory import make
    env = make(item['profile'], images=True)
    obs, _ = env.reset(seed=item['seed'])
    first = np.asarray(obs['rgb'])
    small = np.asarray(item['frame_small'], dtype=float)
    render = first[::16, ::16].astype(float)
    pixel = {'mae_same': float(np.abs(render - small).mean()),
             'mae_bgr_swapped': float(np.abs(render[..., ::-1] - small).mean())}
    state0 = np.asarray(obs['state'], dtype=float)
    info = {}
    for a in item['action']:
        obs, _, t, tr, info = env.step(np.asarray(a, dtype=np.float32))
        if t or tr:
            break
    env.close()
    acts, states = np.asarray(item['action']), np.asarray(item['state'])
    lag = {k: float(np.abs(acts[:len(acts) - k, :5] - states[k:, :5]).mean()) for k in range(0, 8)}
    return {'episode': item['episode'], 'seed': item['seed'], 'replay_outcome': info.get('outcome'),
            'replay_reason': info.get('termination_reason'),
            'state0_matches_recorded': float(np.abs(state0 - states[0]).max()),
            'pixel': pixel, 'action_state_lag_deg': lag, 'best_lag_steps': min(lag, key=lag.get),
            'action_min': acts.min(0).round(1).tolist(), 'action_max': acts.max(0).round(1).tolist(),
            'gripper_values': sorted(set(np.round(acts[:, 5]).astype(int).tolist()))[:6] + ['...'],
            'steps': len(acts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='rosmaster_floor_pick_knowledge_sim')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--replay-json', type=Path)
    ap.add_argument('--out', type=Path, default=DATA / 'diagnosis/e3_replay')
    a = ap.parse_args()
    if a.replay_json:  # MuJoCo side (.venv-sim)
        items = json.loads(a.replay_json.read_text())
        print(json.dumps([replay(i) for i in items]))
        return
    # Only the v4 training pairs (seeds 60106..60405); older 50xxx captures in the same folder are another geometry.
    folders = sorted(p for p in (DATA / 'datasets/sim' / a.dataset).iterdir() if (p / 'provenance.json').exists()
                     and 60106 <= json.loads((p / 'provenance.json').read_text())['hardware']['seed'] < 60406
                     and json.loads((p / 'provenance.json').read_text()).get('outcome') == 'success')
    random.Random(0).shuffle(folders)
    a.out.mkdir(parents=True, exist_ok=True)
    items = [extract(f) for f in folders[:a.episodes]]
    src = a.out / 'items.json'
    src.write_text(json.dumps(items))
    res = subprocess.run([str(ROOT / '.venv-sim/bin/python'), __file__, '--replay-json', str(src)],
                         capture_output=True, text=True, env={'MUJOCO_GL': 'egl', 'PATH': '/usr/bin:/bin',
                                                              'HOME': str(Path.home())}, cwd=ROOT)
    if res.returncode:
        raise SystemExit(res.stderr[-3000:])
    results = json.loads(res.stdout.strip().splitlines()[-1])
    ok = sum(r['replay_outcome'] == 'success' for r in results)
    summary = {'dataset': a.dataset, 'episodes': len(results), 'replay_success': ok, 'results': results}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    src.unlink()
    for r in results:
        print(r['episode'][:8], r['seed'], r['replay_outcome'], 'lag', r['best_lag_steps'],
              'pix %.1f/%.1f' % (r['pixel']['mae_same'], r['pixel']['mae_bgr_swapped']),
              'state0 %.2f' % r['state0_matches_recorded'])
    print('replay success %d/%d' % (ok, len(results)))


if __name__ == '__main__':
    main()
