"""Transfer analysis SO-101 + Tank -> ROSMASTER on FIXTURE decisions in a temporary EXPLORER3_DATA (code path check only).

Generated decisions are marked synthetic=False only inside the temporary data dir, because research.fit refuses to mix
synthetic and physical rows; the report below is labelled domain='fixture' and is not a research result.
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CODE = r'''
import json, sys, time, uuid
import numpy as np
from explorer3 import catalog, research
catalog.initialize(); research.initialize()
rng = np.random.default_rng(7)
N, G = int(sys.argv[2]), int(sys.argv[3])
profiles = {'so101_pick_place': N, 'tank_dock': N, 'rosmaster_pick_place': N}
with catalog.connect() as db:
    for name, count in profiles.items():
        for i in range(count):
            f = {'command_change': float(rng.gamma(2, .3)), 'response_change': float(rng.gamma(2, .2)), 'tracking_error': float(rng.gamma(2, .2)),
                 'observation_age_s': float(rng.uniform(.03, .15)), 'camera_age_s': float(rng.uniform(.05, .2)), 'send_latency_s': None,
                 'progress': None, 'assigned_delay_s': 1.}
            assigned = int(rng.integers(0, 2))
            # Help is useful exactly when tracking error is large (same rule on every robot -> transfer should help).
            p = 1 / (1 + np.exp(-(-.2 - 2.5 * f['tracking_error'] + assigned * (3.5 * f['tracking_error'] - .3))))
            ident = uuid.uuid4().hex
            record = {'episode_id': ident, 'study_id': 'fixture', 'session': f'{name}-s{i % G}', 'scene': f'{name}-scene{i % G}', 'profile': name,
                      'policy_sha256': '0' * 64, 'decision_time_ns': i, 'assigned': assigned, 'propensity': .5, 'assigned_delay_s': 1.,
                      'features': f, 'outcome': 'success' if rng.random() < p else 'failure', 'emergency': False, 'deviation': None,
                      'events': [], 'synthetic': False}
            db.execute('INSERT INTO decisions VALUES(?,?,?,?)', (ident, 'fixture', json.dumps(record), time.time()))
result = research.analyze(sys.argv[1], scope='pair', source_profiles=['so101_pick_place', 'tank_dock'], target_profile='rosmaster_pick_place')
summary = {}
for name, value in result['variants'].items():
    if name.endswith('decision-rules'):
        summary[name] = value; continue
    target = (value.get('report') or {}).get('profiles', {}).get('rosmaster_pick_place', {})
    summary[name] = {'status': value['status'], 'brier_assigned': target.get('brier_assigned'), 'held_out_episodes': target.get('episodes'),
                     'groups': target.get('independent_session_scene_groups'), 'reason': value.get('reason')}
print(json.dumps({'scope': result['scope'], 'profiles': result['profiles'], 'variants': summary}, ensure_ascii=False))
'''


def main():
    size, groups = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('70', '6')
    data = Path(tempfile.mkdtemp(prefix='explorer3-research-fixture-'))
    env = {**os.environ, 'EXPLORER3_DATA': str(data), 'PYTHONPATH': str(ROOT)}
    out = subprocess.check_output([str(ROOT / '.venv/bin/python'), '-c', CODE, str(data / 'analysis'), size, groups], cwd=ROOT, env=env, text=True)
    report = {'fixture_size_per_profile': int(size), 'fixture_groups_per_profile': int(groups), 'domain': 'fixture', 'research_result': False, 'robot_touched': False, 'data': str(data), **json.loads(out.strip().splitlines()[-1])}
    (ROOT / f'docs/handoffs/rosmaster-research-fixture-{size}x{groups}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
