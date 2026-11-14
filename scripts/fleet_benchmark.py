"""Fleet benchmark in simulated rooms with the robots' own code (explorer3.fleet_map / explore /
navigation): how much faster is a team than one robot at mapping a room and at finding an object?

Rooms are random (size, furniture, an internal wall with a doorway); lidars are ray-cast with
noise; ROSMASTER has noisy wheel odometry, Tank none (exactly as on the real robots); the camera
«sees» an object within 2.5 m, ±35°, in line of sight. Time is simulated (0.25 s steps), so the
results do not depend on the machine. Outputs results.jsonl, summary.json and figures.

  .venv-sim/bin/python scripts/fleet_benchmark.py --rooms 20 --workers 6
"""
import argparse
import json
import math
import os
import sys
import time as _time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('EXPLORER3_SIM_BENCHMARK', '1')

CONDITIONS = {'solo': ['rosmaster'], 'team': ['rosmaster', 'tank'], 'team3': ['rosmaster', 'tank', 'rosmaster2']}
DT = .25


class Clock:
    """Simulated time for the modules under test (their time.time/monotonic)."""
    def __init__(self):
        self.t = 1_000_000.

    def time(self):
        return self.t

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def strftime(self, *a):
        return _time.strftime(*a)

    def localtime(self, *a):
        return _time.localtime(*a)

    def time_ns(self):
        return int(self.t * 1e9)

    def monotonic_ns(self):
        return int(self.t * 1e9)


def make_room(rng):
    w, h = rng.uniform(4.5, 7.5), rng.uniform(3.5, 6.)
    segs = [((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))]
    # internal wall with a 0.9 m doorway (two rooms)
    x = rng.uniform(.4, .6) * w
    door = rng.uniform(.8, h - 1.7)
    segs += [((x, 0), (x, door)), ((x, door + .9), (x, h))]
    boxes = []
    for _ in range(rng.integers(3, 7)):
        bw, bh = rng.uniform(.3, 1.2), rng.uniform(.3, .9)
        for _try in range(20):
            bx, by = rng.uniform(.2, w - bw - .2), rng.uniform(.2, h - bh - .2)
            if abs(bx + bw / 2 - x) < bw / 2 + .5 or any(bx < c + d + .6 and c < bx + bw + .6 and by < e + f + .6 and e < by + bh + .6
                                                        for c, e, d, f in boxes):
                continue
            boxes.append((bx, by, bw, bh))
            segs += [((bx, by), (bx + bw, by)), ((bx + bw, by), (bx + bw, by + bh)),
                     ((bx + bw, by + bh), (bx, by + bh)), ((bx, by + bh), (bx, by))]
            break
    return {'w': w, 'h': h, 'segs': np.array(segs, float), 'wall_x': x, 'door': door, 'boxes': boxes}


def wall_distance(room, xy):
    a, b = room['segs'][:, 0], room['segs'][:, 1]
    ab = b - a
    t = np.clip(((xy - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * ab - xy, axis=1).min())


def raycast(room, origin, angles):
    d = np.stack([np.cos(angles), np.sin(angles)], 1)[:, None, :]
    p, e = room['segs'][None, :, 0], room['segs'][None, :, 1] - room['segs'][None, :, 0]
    den = d[..., 0] * -e[..., 1] + d[..., 1] * e[..., 0]
    q = p - np.asarray(origin)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (q[..., 0] * -e[..., 1] + q[..., 1] * e[..., 0]) / den
        u = (d[..., 0] * q[..., 1] - d[..., 1] * q[..., 0]) / den
    return np.where((abs(den) > 1e-9) & (r > .05) & (u >= 0) & (u <= 1), r, np.inf).min(1)


def free_spot(room, rng, clearance=.45, avoid=()):
    for _ in range(500):
        p = np.array([rng.uniform(.4, room['w'] - .4), rng.uniform(.4, room['h'] - .4)])
        if wall_distance(room, p) > clearance and all(np.linalg.norm(p - a) > 1.2 for a in avoid):
            return p
    raise RuntimeError('no free spot')


def reachable_area(room, start, res=.05, clearance=.2):
    """Ground-truth free floor reachable from the start (flood fill), m²."""
    nx, ny = int(room['w'] / res) + 1, int(room['h'] / res) + 1
    xs, ys = (np.arange(nx) + .5) * res, (np.arange(ny) + .5) * res
    free = np.zeros((ny, nx), bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            free[j, i] = wall_distance(room, np.array([x, y])) > clearance
    from collections import deque
    seen = np.zeros_like(free)
    s = (int(start[1] / res), int(start[0] / res))
    q = deque([s])
    seen[s] = True
    while q:
        j, i = q.popleft()
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = j + dj, i + di
            if 0 <= a < ny and 0 <= b < nx and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return seen, res


def run(args):
    room_seed, condition, task = args
    import explorer3.fleet_map as fm
    import explorer3.explore as ex
    import explorer3.navigation as nav
    import explorer3.world_memory as wm
    import explorer3.episodes as eps
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    eps.ROOT = tmp / 'episodes'
    wm.MEMORY = tmp / 'memory.json'
    clock = Clock()
    fm.time = clock
    ex.time = clock
    robots = CONDITIONS[condition]
    for r in robots:                                  # extra robots reuse a real robot's geometry
        kind = 'rosmaster' if r.startswith('rosmaster') else 'tank'
        fm.MOUNTS.setdefault(r, fm.MOUNTS[kind])
        fm.BODY.setdefault(r, fm.BODY[kind])
        for table in (nav.RADIUS, nav.BODY_FRONT, nav.BODY_HALF_W):
            table.setdefault(r, table[kind])
        wm.POSES.setdefault(r, wm.Pose(r))
    fleet = fm.FleetMap()
    fm.FLEET = fleet
    ex.FLEET = fleet
    ex.ACTIVE.clear()
    # Paired design: the room, every robot's start and the hidden object depend only on the room
    # seed, never on the condition — «solo» is the same ROSMASTER start without the helpers.
    rng = np.random.default_rng(room_seed)
    room = make_room(rng)
    all_starts = []
    for _ in range(3):
        all_starts.append(free_spot(room, rng, avoid=all_starts))
    headings = [float(rng.uniform(-math.pi, math.pi)) for _ in range(3)]
    order = {'rosmaster': 0, 'tank': 1, 'rosmaster2': 2}
    starts = [all_starts[order[r]] for r in robots]
    true = {r: (float(all_starts[order[r]][0]), float(all_starts[order[r]][1]), headings[order[r]]) for r in robots}
    odom = {r: tuple(true[r]) for r in robots}
    reach, res = reachable_area(room, starts[0])
    total = reach.sum() * res * res
    obj = None
    if task == 'find':
        orng = np.random.default_rng(room_seed + 7919)
        for min_d in (2.0, 1.5, 1.0):
            for _ in range(300):
                o = free_spot(room, orng, clearance=.3)
                if reach[int(o[1] / res), int(o[0] / res)] and all(np.linalg.norm(o - s) > min_d for s in all_starts):
                    obj = o
                    break
            if obj is not None:
                break
        if obj is None:
            return {'room': room_seed, 'condition': condition, 'task': task, 'skipped': 'no place for the object'}
    found = {'by': None, 't': None}

    def stop_when():
        return 'found' if found['by'] else None

    tasks = {r: ex.Explore(r, record=False, time_limit_s=900, stop_when=stop_when if task == 'find' else None) for r in robots}
    for r in robots:
        ex.ACTIVE[r] = tasks[r]
    curve, t0, collisions = [], clock.t, 0
    origin_pose = None
    for step in range(int(900 / DT)):
        for r in robots:
            x, y, th = true[r]
            arrays, streams = {}, {}
            for name, (mx, my, myaw, direction) in fm.MOUNTS[r].items():
                o = (x + math.cos(th) * mx - math.sin(th) * my, y + math.sin(th) * mx + math.cos(th) * my)
                ang = th + myaw + direction * 2 * math.pi * np.arange(450) / 450
                rr = raycast(room, o, ang)
                rr = np.where(rr < 12, rr + rng.normal(0, .01, 450), np.inf)
                arrays[name] = rr.astype(np.float32)
                streams[name] = {'angle_min': 0., 'angle_increment': 2 * math.pi / 450, 'range_min': .05}
            fleet.update(r, {'arrays': arrays, 'meta': {'streams': streams}}, odom[r] if r.startswith('rosmaster') else None)
            if origin_pose is None:
                origin_pose = (x, y, th)
            if task == 'find' and not found['by']:
                d = obj - np.array([x, y])
                dist, bearing = float(np.linalg.norm(d)), math.atan2(d[1], d[0]) - th
                if dist < 2.5 and abs(fm.wrap(bearing)) < math.radians(35) and raycast(room, (x, y), np.array([math.atan2(d[1], d[0])]))[0] > dist - .05:
                    found.update(by=r, t=clock.t - t0)
        for r in robots:
            v, w = tasks[r].command(None) if tasks[r].active else (0., 0.)
            x, y, th = true[r]
            nx_, ny_, nth = x + v * DT * math.cos(th), y + v * DT * math.sin(th), fm.wrap(th + w * DT)
            if wall_distance(room, np.array([nx_, ny_])) < fm.BODY[r][3] - .01:
                collisions += 1                               # blocked by the wall: does not move
                nx_, ny_ = x, y
            true[r] = (nx_, ny_, nth)
            ox, oy, oth = odom[r]                              # wheel odometry with 3 % / 2 % error
            ov, ow = v * (1 + rng.normal(0, .03)), w * (1 + rng.normal(0, .02))
            odom[r] = (ox + ov * DT * math.cos(oth), oy + ov * DT * math.sin(oth), fm.wrap(oth + ow * DT))
        clock.t += DT
        if step % 8 == 0:                                      # coverage every 2 s
            free = fleet.grid.known_free()
            ys, xs = np.nonzero(free)
            pts = fm.transform(np.stack([(xs - fm.SIZE // 2 + .5) * fm.RES, (ys - fm.SIZE // 2 + .5) * fm.RES], 1), origin_pose)
            gi, gj = (pts[:, 0] / res).astype(int), (pts[:, 1] / res).astype(int)
            ok = (gi >= 0) & (gj >= 0) & (gi < reach.shape[1]) & (gj < reach.shape[0])
            covered = np.zeros_like(reach)
            covered[gj[ok], gi[ok]] = True
            import cv2                                     # a rotated 5 cm grid leaves moire holes: close them
            covered = cv2.dilate(covered.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            curve.append((round(clock.t - t0, 1), round(float((covered & reach).sum() / max(1, reach.sum())), 3)))
        if all(not tasks[r].active for r in robots):
            break
    t90 = next((t for t, c in curve if c >= .9), None)
    t80 = next((t for t, c in curve if c >= .8), None)
    states = {r: fleet.track.get(r, {}).get('state') for r in robots}
    return {'room': room_seed, 'condition': condition, 'task': task, 'robots': robots, 'area_m2': round(float(total), 1),
            'seconds': round(clock.t - t0, 1), 't90': t90, 't80': t80, 'final_coverage': curve[-1][1] if curve else 0,
            'found_by': found['by'], 'found_t': found['t'], 'collisions': collisions,
            'contrib': dict(fleet.contrib), 'final_states': states,
            'outcomes': {r: tasks[r].state for r in robots}, 'curve': curve}


def safe_run(args):
    try:
        return run(args)
    except Exception as exc:
        return {'room': args[0], 'condition': args[1], 'task': args[2], 'skipped': f'{type(exc).__name__}: {exc}'}


def summarize(results, out):
    import statistics as st
    results = [r for r in results if 'skipped' not in r]
    summary = {}
    for task in ('explore', 'find'):
        for cond in CONDITIONS:
            rows = [r for r in results if r['task'] == task and r['condition'] == cond]
            if not rows:
                continue
            if task == 'explore':
                t = [r['t90'] for r in rows if r['t90'] is not None]
                t8 = [r['t80'] for r in rows if r.get('t80') is not None]
                summary[f'{task}/{cond}'] = {'runs': len(rows), 'reached_90pct': len(t), 'median_t90_s': st.median(t) if t else None,
                                             'reached_80pct': len(t8), 'median_t80_s': st.median(t8) if t8 else None,
                                             'mean_final_coverage': round(st.mean(r['final_coverage'] for r in rows), 3),
                                             'collisions_per_run': round(st.mean(r['collisions'] for r in rows), 2)}
            else:
                t = [r['found_t'] for r in rows if r['found_t'] is not None]
                summary[f'{task}/{cond}'] = {'runs': len(rows), 'found': len(t), 'median_find_s': st.median(t) if t else None,
                                             'mean_find_s': round(st.mean(t), 1) if t else None}
    for cond in ('team', 'team3'):
        pairs = [(a['t80'], b['t80']) for a in results for b in results
                 if a['task'] == b['task'] == 'explore' and a['room'] == b['room'] and a['condition'] == 'solo' and b['condition'] == cond
                 and a.get('t80') and b.get('t80')]
        if pairs:
            summary[f'speedup_explore/{cond}'] = round(float(np.median([x / y for x, y in pairs])), 2)
        pairs = [(a['found_t'], b['found_t']) for a in results for b in results
                 if a['task'] == b['task'] == 'find' and a['room'] == b['room'] and a['condition'] == 'solo' and b['condition'] == cond
                 and a['found_t'] and b['found_t']]
        if pairs:
            summary[f'speedup_find/{cond}'] = round(float(np.median([x / y for x, y in pairs])), 2)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        grid = np.arange(0, 901, 10)
        names = {'solo': 'ROSMASTER один', 'team': 'ROSMASTER + Tank', 'team3': '3 робота'}
        for cond in CONDITIONS:
            rows = [r for r in results if r['task'] == 'explore' and r['condition'] == cond]
            if not rows:
                continue
            curves = np.array([np.interp(grid, [t for t, _ in r['curve']], [c for _, c in r['curve']]) for r in rows])
            m = np.median(curves, 0)
            ax[0].plot(grid, m, label=names[cond])
            ax[0].fill_between(grid, np.percentile(curves, 25, 0), np.percentile(curves, 75, 0), alpha=.2)
            found = sorted(r['found_t'] if r['found_t'] is not None else np.inf for r in results if r['task'] == 'find' and r['condition'] == cond)
            if found:
                ax[1].step([0] + [f for f in found if np.isfinite(f)], np.arange(0, sum(np.isfinite(found)) + 1) / len(found), where='post', label=names[cond])
        ax[0].set(xlabel='время, с', ylabel='доля комнаты на общей карте', title='Осмотр комнаты', xlim=(0, 600))
        ax[1].set(xlabel='время, с', ylabel='доля найденных', title='Поиск предмета', xlim=(0, 600))
        for a in ax:
            a.grid(alpha=.3)
            a.legend()
        fig.tight_layout()
        fig.savefig(out / 'fleet_benchmark.png', dpi=130)
    except Exception as exc:
        print('plot failed:', exc)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rooms', type=int, default=20)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--conditions', default='solo,team,team3')
    ap.add_argument('--tasks', default='explore,find')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    data = Path(os.environ.get('EXPLORER3_DATA', Path.home() / '.local/share/explorer3'))
    out = Path(a.out) if a.out else data / 'simlab' / 'fleet_benchmark' / _time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(1000 + i, c, t) for i in range(a.rooms) for c in a.conditions.split(',') for t in a.tasks.split(',')]
    results = []
    from multiprocessing import Pool
    with Pool(a.workers) as pool, (out / 'results.jsonl').open('w') as f:
        for i, r in enumerate(pool.imap_unordered(safe_run, jobs)):
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            if 'skipped' in r:
                print(f"[{i + 1}/{len(jobs)}] room {r['room']} {r['condition']} {r['task']} SKIPPED: {r['skipped']}", flush=True)
                continue
            print(f"[{i + 1}/{len(jobs)}] room {r['room']} {r['condition']:5} {r['task']:7} t90={r['t90']} found={r['found_t']} "
                  f"cov={r['final_coverage']} coll={r['collisions']} states={r['final_states']}", flush=True)
    print(json.dumps(summarize(results, out), indent=1, ensure_ascii=False))
    print('out:', out)


if __name__ == '__main__':
    main()
