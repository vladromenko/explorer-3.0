"""Fit the MuJoCo task scenes to the real demonstrations (blocks 3 and 4 of DATASET_RECORDING_RU.md).

The recorded joint angles go through the same MuJoCo arm model (forward kinematics), so we measure
where the real arm started, where it closed the gripper on the object and where it opened it for
the person. Those numbers replace the instruction defaults in configs/sim-alignment.json, so the
next simulated demonstrations, training rehearsal and MuJoCo checks use the user's real layout.

  .venv-sim/bin/python scripts/sim_align.py            # report only
  .venv-sim/bin/python scripts/sim_align.py --apply    # write configs/sim-alignment.json
Needs at least 5 successful demonstrations per block; otherwise the defaults stay.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MIN_DEMOS = 5


def demos(profile, words):
    """Successful, not excluded real demonstrations whose series is this block."""
    from explorer3 import catalog
    with catalog.connect() as db:
        series = {r[0]: (r[1] or '', r[2] or '') for r in db.execute('SELECT id, description, instruction FROM collection_series')}
    out = []
    for e in catalog.rows('episodes'):
        if e['profile'] != profile or e['outcome'] != 'success' or e.get('excluded'):
            continue
        text = ' '.join(series.get(e['session'], ('', ''))).lower()
        if any(w in text for w in words):
            rows = [json.loads(line) for line in (Path(e['dataset']) / 'research.jsonl').read_text().splitlines()]
            out.append((e['id'], np.array([r['observation.state'][:6] for r in rows], float),
                        np.array([r['action'][:6] for r in rows], float)))
    return out


def fk(env, state):
    import mujoco
    d = mujoco.MjData(env.model)
    d.qpos[:6] = env.to_radians(state)
    mujoco.mj_kinematics(env.model, d)
    return d.site('gripperframe').xpos.copy()


def so101(fps=30):
    from explorer3.sim.aligned import SO101Handover
    env = SO101Handover()
    found = demos('so101_pick_place', ('передач', 'hand over'))
    starts, grasps, gives = [], [], []
    for ident, S, A in found:
        open_level = float(np.median(A[:10, 5]))
        closing = np.nonzero(A[:, 5] < open_level - 8)[0]
        if not len(closing):
            continue
        g = min(len(S) - 1, int(closing[0]) + fps // 2)
        hold = float(np.median(A[g:g + fps, 5]))
        opening = [i for i in range(g + fps, len(A)) if A[i, 5] > hold + 8]
        starts.append(S[0])
        grasps.append(fk(env, S[g]))
        if opening:
            gives.append(fk(env, S[opening[0]]))
    return found, starts, grasps, gives


def rosmaster(fps=20):
    from explorer3.sim.aligned import RosmasterFloorPick
    env = RosmasterFloorPick()
    found = demos('rosmaster_pick_place', ('взять с пола', 'pick up'))
    starts, grasps = [], []
    for ident, S, A in found:
        closing = np.nonzero(A[:, 5] > A[0, 5] + 20)[0]
        if not len(closing):
            continue
        starts.append(S[0])
        grasps.append(fk(env, S[min(len(S) - 1, int(closing[0]) + fps // 2)]))
    return found, starts, grasps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    path = ROOT / 'configs/sim-alignment.json'
    cfg = json.loads(path.read_text())
    report = {'at': time.strftime('%Y-%m-%d %H:%M'), 'so101_handover': {}, 'rosmaster_floor_pick': {}}
    stamp = time.strftime('%Y-%m-%d')

    found, starts, grasps, gives = so101()
    r = report['so101_handover']
    r.update(demos=len(found), with_grasp=len(grasps), with_handover=len(gives))
    if len(grasps) >= MIN_DEMOS and len(gives) >= MIN_DEMOS:
        G, H, S0 = np.array(grasps), np.array(gives), np.array(starts)
        spread = np.percentile(G[:, :2], 90, 0) - np.percentile(G[:, :2], 10, 0)
        new = {'rest_pose': np.round(np.median(S0, 0), 1).tolist(),
               'pad_center': np.round(np.median(G[:, :2], 0), 3).tolist(),
               'pad_zone_step_m': round(float(max(.01, spread.max() / 2)), 3),
               'handover_point': np.round(np.median(H, 0), 3).tolist(),
               'pad_height_m': round(float(np.median(G[:, 2])), 3)}
        r.update(measured=new, previous={k: cfg['so101_handover'].get(k) for k in new})
        if new['pad_height_m'] > .03:
            r['warning'] = 'Площадка выше стола на {:.0f} см: в MuJoCo пока стол без подставки'.format(new['pad_height_m'] * 100)
        new.pop('pad_height_m')
        if a.apply:
            cfg['so101_handover'].update(new, source=f'measured: {len(grasps)} real demonstrations, {stamp}')
    else:
        r['kept'] = f'нужно ≥{MIN_DEMOS} успешных показов блока 4 («передача: …»)'

    found, starts, grasps = rosmaster()
    r = report['rosmaster_floor_pick']
    r.update(demos=len(found), with_grasp=len(grasps))
    if len(grasps) >= MIN_DEMOS:
        G, S0 = np.array(grasps), np.array(starts)
        new = {'start_pose': np.round(np.median(S0, 0), 1).tolist(),
               'object_x_m': np.round([np.percentile(G[:, 0], 5), np.percentile(G[:, 0], 95)], 3).tolist(),
               'object_y_m': np.round([np.percentile(G[:, 1], 5), np.percentile(G[:, 1], 95)], 3).tolist()}
        r.update(measured=new, previous={k: cfg['rosmaster_floor_pick'].get(k) for k in new})
        if a.apply:
            cfg['rosmaster_floor_pick'].update(new, source=f'measured: {len(grasps)} real demonstrations, {stamp}')
    else:
        r['kept'] = f'нужно ≥{MIN_DEMOS} успешных показов блока 3 («взять с пола: …»)'

    if a.apply:
        path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
