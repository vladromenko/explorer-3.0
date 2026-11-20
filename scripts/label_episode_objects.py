"""Offline object labels for recorded demos (run AFTER a recording session, never during it).

For every successful attempt of a task, runs SAM3 on the first wrist-camera frame with the
series object name (one English noun: cube / candy / toothpaste) and writes
DATA/validation/collection/<episode>/object.json: name, box, score, overlay.
Detection does not run during recording on purpose: GPU load there caused frame gaps.

  .venv/bin/python scripts/label_episode_objects.py [--task rosmaster_pick_place] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explorer3 import catalog  # noqa: E402
from explorer3.paths import DATA  # noqa: E402
from explorer3.perception import analyze  # noqa: E402

# Series descriptions/instructions -> single noun SAM3 recognises (tested 20.11).
NAMES = {'кубик': 'cube', 'cube': 'cube', 'candy': 'candy', 'конфета': 'candy',
         'tooth paste': 'toothpaste', 'toothpaste': 'toothpaste', 'паста': 'toothpaste'}


def object_name(series):
    text = (series['description'] + ' ' + (series['instruction'] or '')).lower()
    for key, name in NAMES.items():
        if key in text:
            return name
    words = (series['instruction'] or '').lower().replace('pick up the', '').split()
    return words[-1] if words else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='rosmaster_pick_place')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    with catalog.connect() as db:
        series = {r['id']: dict(r) for r in db.execute('SELECT * FROM collection_series WHERE task=?', (args.task,))}
        attempts = [dict(r) for r in db.execute("SELECT * FROM collection_attempts WHERE outcome='success'")]
    report = []
    for a in attempts:
        if a['series'] not in series or not a['episode']:
            continue
        folder = DATA / 'validation/collection' / a['episode']
        out = folder / 'object.json'
        if out.exists() and not args.force:
            report.append(json.loads(out.read_text()))
            continue
        name = object_name(series[a['series']])
        frame = folder / 'first.jpg'
        if not name or not frame.exists():
            continue
        r = analyze(frame, folder / 'object', 'sam3', name, device='cuda')
        best = max(range(len(r['scores'])), key=lambda i: r['scores'][i]) if r['scores'] else None
        label = {'episode': a['episode'], 'series': a['series'], 'object': name,
                 'found': best is not None, 'score': r['scores'][best] if best is not None else None,
                 'box_xyxy': r['boxes_xyxy'][best] if best is not None else None,
                 'image_size': r['image_size'], 'method': 'sam3', 'overlay': r['overlay']}
        out.write_text(json.dumps(label, ensure_ascii=False, indent=1))
        report.append(label)
    for x in report:
        print(x['series'][-3:], x['episode'][:12], x['object'], 'FOUND %.2f' % x['score'] if x['found'] else 'NOT FOUND')


if __name__ == '__main__':
    main()
