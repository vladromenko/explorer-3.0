"""Statistics for the article from a fleet_benchmark result folder: 95% intervals and paired tests.

  .venv/bin/python scripts/benchmark_stats.py docs/results/fleet_benchmark_run30_visual

- success rate: Wilson 95% interval;
- median time: bootstrap 95% interval (10 000 resamples, rooms resampled);
- speed-up team vs solo: per-room ratio solo/team on the same room and start (paired design),
  median with a bootstrap interval; a run that did not finish counts with its full run time
  (a conservative lower bound for the solo time, an upper bound for the team time is not needed:
  unfinished team runs make the team look slower, never faster);
- Wilcoxon signed-rank test on the paired log-times.
Writes stats.json next to summary.json and prints a table.
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0., c - h), 3), round(min(1., c + h), 3)]


def boot_median(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return None
    idx = RNG.integers(0, len(x), (n, len(x)))
    m = np.median(x[idx], axis=1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


def load(folder):
    rows = [json.loads(line) for line in gzip.open(Path(folder) / 'results.jsonl.gz', 'rt')]
    return [r for r in rows if 'skipped' not in r]


def time_of(r):
    """Finish time; an unfinished run counts with its whole duration (censored)."""
    if r['task'] == 'find':
        return (r['found_t'], True) if r.get('found_t') is not None else (r['seconds'], False)
    return (r['t80'], True) if r.get('t80') is not None else (r['seconds'], False)


def stats(rows):
    from scipy.stats import wilcoxon
    out = {}
    for task in sorted({r['task'] for r in rows}):
        conds = sorted({r['condition'] for r in rows if r['task'] == task})
        for cond in conds:
            sel = [r for r in rows if r['task'] == task and r['condition'] == cond]
            done = [time_of(r)[0] for r in sel if time_of(r)[1]]
            out[f'{task}/{cond}'] = {
                'runs': len(sel), 'finished': len(done), 'finished_ci95': wilson(len(done), len(sel)),
                'median_s': round(float(np.median(done)), 2) if done else None, 'median_s_ci95': boot_median(done),
                'time': 't80 (80% of the room)' if task == 'explore' else 'object found',
                'collisions': int(sum(r.get('collisions', 0) for r in sel))}
        solo = {(r['room']): r for r in rows if r['task'] == task and r['condition'] == 'solo'}
        for cond in conds:
            if cond == 'solo':
                continue
            pairs = [(solo[r['room']], r) for r in rows if r['task'] == task and r['condition'] == cond and r['room'] in solo]
            if not pairs:
                continue
            a = np.array([time_of(s)[0] for s, _ in pairs], float)
            b = np.array([time_of(t)[0] for _, t in pairs], float)
            a, b = np.maximum(a, .25), np.maximum(b, .25)      # seen at the start: one control tick
            ok = np.ones(len(a), bool)
            ratio = a[ok] / b[ok]
            entry = {'pairs': int(ok.sum()), 'median_speedup': round(float(np.median(ratio)), 2),
                     'median_speedup_ci95': boot_median(ratio), 'team_faster_in': int((ratio > 1).sum())}
            diff = np.log(a[ok]) - np.log(b[ok])
            if (diff != 0).sum() >= 6:
                entry['wilcoxon_p'] = float(f'{wilcoxon(diff).pvalue:.2g}')
            out[f'speedup_{task}/{cond}'] = entry
    return out


def table(s):
    lines = []
    for k, v in s.items():
        if k.startswith('speedup'):
            lines.append(f"{k:24s} ×{v['median_speedup']} [{v['median_speedup_ci95']}]  "
                         f"быстрее в {v['team_faster_in']}/{v['pairs']}  p={v.get('wilcoxon_p')}")
        else:
            lines.append(f"{k:24s} {v['finished']}/{v['runs']} {v['finished_ci95']}  "
                         f"медиана {v['median_s']} с {v['median_s_ci95']}  столкновений {v['collisions']}")
    return '\n'.join(lines)


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    s = stats(load(folder))
    (folder / 'stats.json').write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(table(s))
