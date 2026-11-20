"""Gate check of a recorded ROSMASTER series (run between series, never while recording).

For every successful attempt of the series: recorder audit (gaps <= 200 ms, camera continuity,
action/state match), lift >= 5 cm held >= 2 s (gripper-point height from the measured joints via
the arm model), optional object found on the first frame (SAM3, GPU ~1 min).
Prints a short Russian verdict per episode and for the series.

  .venv/bin/python scripts/check_series.py                 # latest series
  .venv/bin/python scripts/check_series.py "cube, обучение" --label
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3 import catalog  # noqa: E402
from explorer3.paths import DATA  # noqa: E402


def lift(root):
    from explorer3.sim.vendor_rosmaster.arm_model import ArmModel
    rows = [json.loads(line) for line in (root / 'research.jsonl').read_text().splitlines() if line.strip()]
    st = np.array([r['observation.state'] for r in rows], float)
    t = np.array([r['observation_timestamp_ns'] for r in rows]) / 1e9
    z = np.array([ArmModel().tcp_position(s[:5])[2] for s in st])
    i = int(z.argmin())
    up = (z[i:] - z[i]) >= .05
    best = run = 0.
    for k in range(1, len(up)):
        run = run + (t[i + k] - t[i + k - 1]) if up[k] and up[k - 1] else 0.
        best = max(best, run)
    return {'lift_cm': round(float((z[i:] - z[i]).max()) * 100, 1), 'held_s': round(best, 1),
            'duration_s': round(float(t[-1] - t[0]), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('series', nargs='?', help='description (substring); default: latest series')
    ap.add_argument('--label', action='store_true', help='also SAM3 object check on the first frame (GPU)')
    a = ap.parse_args()
    with catalog.connect() as db:
        if db.execute('SELECT 1 FROM collection_attempts WHERE outcome IS NULL AND created>?',
                      (time.time() - 3600,)).fetchone():
            raise SystemExit('Идёт запись — проверку запускаю после серии')
        q = 'SELECT * FROM collection_series WHERE task=?' + (' AND description LIKE ?' if a.series else '') + ' ORDER BY created DESC LIMIT 1'
        s = db.execute(q, ('rosmaster_pick_place', '%' + a.series + '%') if a.series else ('rosmaster_pick_place',)).fetchone()
        if s is None:
            raise SystemExit('Серия не найдена')
        s = dict(s)
        attempts = [dict(r) for r in db.execute('SELECT * FROM collection_attempts WHERE series=? ORDER BY created', (s['id'],))]
    if a.label:
        subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/label_episode_objects.py')],
                       cwd=ROOT, check=False, capture_output=True)
    print('Серия: %s  (%s)' % (s['description'], s['instruction']))
    good = 0
    for n, at in enumerate(attempts, 1):
        if at['outcome'] != 'success':
            print('%2d  %-8s —' % (n, at['outcome']))
            continue
        quality = json.loads(at['quality'] or '{}')
        problems = list(quality.get('problems', []))
        if quality.get('status') != 'pass' and not problems:
            problems.append('аудит ещё не завершён')
        root = DATA / 'datasets/rosmaster_pick_place' / at['episode']
        m = lift(root) if (root / 'research.jsonl').exists() else {'lift_cm': 0, 'held_s': 0, 'duration_s': 0}
        if m['lift_cm'] < 5 or m['held_s'] < 2:
            problems.append('подъём %.1f см, удержание %.1f с (нужно ≥5 см и ≥2 с)' % (m['lift_cm'], m['held_s']))
        obj = DATA / 'validation/collection' / at['episode'] / 'object.json'
        label = json.loads(obj.read_text()) if obj.exists() else None
        if label is not None and not label['found']:
            problems.append('предмет «%s» не найден на первом кадре' % label['object'])
        ok = not problems
        good += ok
        print('%2d  %s  %s  подъём %.1f см / %.1f с, %.0f с%s  %s' % (
            n, 'ГОДЕН ' if ok else 'ПЕРЕСНЯТЬ', at['episode'][:8], m['lift_cm'], m['held_s'], m['duration_s'],
            '' if label is None else ', предмет %.2f' % (label['score'] or 0), '; '.join(problems)))
    print('Итого годных: %d из %d попыток' % (good, len(attempts)))


if __name__ == '__main__':
    main()
