"""Timing/tracking part of the transfer error budget (review item 9), from recorded real episodes.

Per successful ROSMASTER episode (rosmaster_pick_place) and over all of them:
- frame period and gaps (observation_timestamp_ns);
- camera age at observation: observation time - camera frame time (both Homebrain clock);
- command availability after observation (action_available_ns - observation_timestamp_ns);
- servo tracking: lag (in 50 ms steps) that best aligns action[t] with measured state[t+k],
  and the residual tracking error per joint in degrees at that lag.
Writes docs/results/latency_budget/summary.json (no images, numbers only).

  .venv/bin/python scripts/latency_budget.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3 import catalog  # noqa: E402
from explorer3.paths import DATA  # noqa: E402

JOINTS = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']


def pct(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {'p50': round(float(np.percentile(x, 50)), 1), 'p95': round(float(np.percentile(x, 95)), 1),
            'max': round(float(x.max()), 1)} if len(x) else None


def episode(folder):
    rows = [json.loads(line) for line in (folder / 'research.jsonl').read_text().splitlines() if line.strip()]
    obs = np.array([r['observation_timestamp_ns'] for r in rows], float)
    period = np.diff(obs) / 1e6
    cam_age = [(r['observation_timestamp_ns'] - r['camera']['timestamp_ns']) / 1e6 for r in rows
               if isinstance(r.get('camera'), dict) and r['camera'].get('timestamp_ns')]
    avail = [(r['action_available_ns'] - r['observation_timestamp_ns']) / 1e6 for r in rows if r.get('action_available_ns')]
    act = np.array([r['action'] for r in rows], float)
    st = np.array([r['observation.state'] for r in rows], float)
    lags = {k: float(np.abs(act[:len(act) - k] - st[k:]).mean()) for k in range(0, 11)}
    best = min(lags, key=lags.get)
    err = np.abs(act[:len(act) - best] - st[best:])
    return {'episode': folder.name, 'frames': len(rows), 'period_ms': pct(period), 'gaps_over_200ms': int((period > 200).sum()),
            'camera_age_ms': pct(cam_age), 'action_after_observation_ms': pct(avail),
            'best_lag_steps': best, 'tracking_err_deg_at_lag': {j: round(float(e), 2) for j, e in zip(JOINTS, err.mean(0))},
            'tracking_err_deg_p95_at_lag': {j: round(float(e), 2) for j, e in zip(JOINTS, np.percentile(err, 95, axis=0))},
            '_period': period.tolist(), '_cam': cam_age, '_avail': avail, '_err': err.tolist()}


def main():
    with catalog.connect() as db:
        eps = [r[0] for r in db.execute(
            "SELECT a.episode FROM collection_attempts a JOIN collection_series s ON a.series=s.id "
            "WHERE s.task='rosmaster_pick_place' AND a.outcome='success' AND a.episode IS NOT NULL")]
    per = [episode(DATA / 'datasets/rosmaster_pick_place' / e) for e in eps
           if (DATA / 'datasets/rosmaster_pick_place' / e / 'research.jsonl').exists()]
    err = np.concatenate([np.array(p['_err']) for p in per])
    total = {'episodes': len(per),
             'period_ms': pct(sum((p['_period'] for p in per), [])),
             'camera_age_ms': pct(sum((p['_cam'] for p in per), [])),
             'action_after_observation_ms': pct(sum((p['_avail'] for p in per), [])),
             'best_lag_steps': pct([p['best_lag_steps'] for p in per]),
             'tracking_err_deg_mean': {j: round(float(e), 2) for j, e in zip(JOINTS, err.mean(0))},
             'tracking_err_deg_p95': {j: round(float(e), 2) for j, e in zip(JOINTS, np.percentile(err, 95, axis=0))}}
    for p in per:
        for k in [k for k in p if k.startswith('_')]:
            del p[k]
    out = ROOT / 'docs/results/latency_budget'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps({'total': total, 'episodes': per}, indent=1))
    print(json.dumps(total, indent=1))


if __name__ == '__main__':
    main()
