"""ROSMASTER response identification from recorded REAL episodes (no robot motion, no training).

What is fitted, per channel, against what the controller actually received and what was measured:
  base  (rosmaster_dock):       sent velocity [vx, vy, wz] -> encoder velocity      y' = (gain*u(t-d) - y)/tau
  arm   (rosmaster_pick_place): sent servo target [j1..j5] -> servo readback      y' = clip((u(t-d) - y)/tau, +-vmax)
The simulator itself uses the same first-order form without dead time / rate limit (mobile.py lag_s+slip, environment.py lag_s);
fitting both forms shows whether that simpler structure is enough.

Timing: command time = agent `*_sent_ns`; measurement time = next agent snapshot minus each channel's reported sensor age.
Frames are 20 Hz, so time constants below ~50 ms are poorly resolved. Connected session/scene groups stay together;
the most recent groups are held out. The candidate is judged ONLY on held-out episodes, never applied automatically.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

DT = .01
DELAYS = np.round(np.arange(0., .32, .02), 3)
TAUS = np.round(np.geomspace(.02, .8, 14), 4)
VMAX = np.array([30., 60., 90., 120., 180., 270., np.inf])          # deg/s, arm only
NOMINAL = {'rosmaster_dock': {'delay_s': 0., 'tau_s': .10, 'gain': .925, 'source': 'explorer3/sim/mobile.py reset(): lag U(.06,.14), slip U(.85,1)'},
           'rosmaster_pick_place': {'delay_s': 0., 'tau_s': .0425, 'gain': 1., 'vmax_deg_s': None,
                                    'source': 'explorer3/sim/environment.py reset(): lag U(.025,.06) (SO-101 value)'}}
CHANNELS = {'rosmaster_dock': ['vx_m_s', 'vy_m_s', 'wz_rad_s'], 'rosmaster_pick_place': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']}


def series(rows, profile):
    """(command times, command values, measurement times, measured values), seconds on the agent clock."""
    base = profile == 'rosmaster_dock'
    k = 3 if base else 5
    ct, cv, mt, mv = [], [], [], []
    for row, following in zip(rows, rows[1:]):
        if row.get('action_sent') is not None and row.get('action_sent_agent_ns') is not None:
            ct.append(row['action_sent_agent_ns'] / 1e9); cv.append([float(v) for v in row['action_sent'][:k]])
        measured, age = row.get('measured_response'), row.get('measured_response_age_s')
        if measured is None or age is None:
            continue
        ages=list(age[:k]) if isinstance(age,list) else [age]*k
        if len(ages)!=k or any(v is None or not math.isfinite(v) or v<0 for v in ages):
            raise ValueError('Invalid measured feedback age')
        if not row.get('controller_ready', True) or max(ages) > .5:
            continue
        if len(measured)<k:raise ValueError('Incomplete measured response')
        mt.append([following['agent_monotonic_ns'] / 1e9 - a for a in ages]); mv.append([float(v) for v in measured[:k]])
    if len(ct) < 5 or len(mt) < 10:
        return None
    order = np.argsort(ct)
    result=np.array(ct)[order],np.array(cv)[order],np.array(mt),np.array(mv)
    if result[1].shape!=(len(ct),k) or not all(np.isfinite(x).all() for x in result):raise ValueError('Invalid command/response series')
    if (np.diff(result[2],axis=0)<-1e-8).any():raise ValueError('Measured timestamps moved backwards')
    return result


def simulate(ct, cv, mt, y0, delays, taus, gains, vmax):
    """Vectorized over candidates. Returns predictions (P, len(mt)). Before the first command the output holds y0."""
    t0, t1 = mt[0], mt[-1]
    grid = np.arange(t0, t1 + DT, DT)
    unique = np.unique(delays)
    command = {}
    for d in unique:
        index = np.searchsorted(ct, grid - d, side='right') - 1
        u = np.where(index >= 0, cv[np.clip(index, 0, None)], np.nan)
        command[d] = u
    P = len(delays)
    y = np.full(P, y0, dtype=float)
    out = np.empty((P, len(grid)))
    alpha = np.minimum(1., DT / taus)
    for i in range(len(grid)):
        u = np.empty(P)
        for d in unique:
            mask = delays == d
            u[mask] = command[d][i]
        target = np.where(np.isnan(u), y / np.where(gains == 0, 1, gains), u) * gains
        step = alpha * (target - y)
        y = y + np.clip(step, -vmax * DT, vmax * DT)
        out[:, i] = y
    at = np.clip(np.round((mt - t0) / DT).astype(int), 0, len(grid) - 1)
    return out[:, at]


def candidates(profile, simple):
    if simple:
        delays, taus = np.zeros(len(TAUS)), TAUS.copy()
        vmax = np.full(len(TAUS), np.inf)
    else:
        rates = VMAX if profile == 'rosmaster_pick_place' else np.array([np.inf])
        mesh = np.array(np.meshgrid(DELAYS, TAUS, rates, indexing='ij')).reshape(3, -1)
        delays, taus, vmax = mesh
    return delays, taus, vmax


def fit_channel(train, profile, channel, simple=False):
    """Grid over (delay, tau[, vmax]); gain from least squares for the base, 1 for servo positions."""
    base = profile == 'rosmaster_dock'
    delays, taus, vmax = candidates(profile, simple)
    sse = np.zeros(len(delays)); num = np.zeros(len(delays)); den = np.zeros(len(delays)); n = 0
    if base:
        # y = decay(y0) + gain*G  (linear in gain): run with gain=1 from y0=0, the y0 term is the free response.
        parts = []
        for ct, cv, mt, mv in train:
            mt=mt[:,channel] if mt.ndim==2 else mt
            ones = np.ones(len(delays))
            g = simulate(ct, cv[:, channel], mt, 0., delays, taus, ones, vmax)
            free = np.exp(-(mt - mt[0])[None, :] / taus[:, None]) * mv[0, channel]
            residual_target = mv[:, channel][None, :] - free
            num += (g * residual_target).sum(1); den += (g * g).sum(1)
            parts.append((g, residual_target))
        gains = np.clip(num / np.maximum(den, 1e-12), .3, 1.5)
        for g, target in parts:
            sse += ((target - gains[:, None] * g) ** 2).sum(1); n += target.shape[1]
    else:
        gains = np.ones(len(delays))
        for ct, cv, mt, mv in train:
            mt=mt[:,channel] if mt.ndim==2 else mt
            pred = simulate(ct, cv[:, channel], mt, mv[0, channel], delays, taus, gains, vmax)
            sse += ((mv[:, channel][None, :] - pred) ** 2).sum(1); n += len(mt)
    best = int(np.argmin(sse))
    result = {'delay_s': float(delays[best]), 'tau_s': float(taus[best]), 'gain': round(float(gains[best]), 4),
              'fit_rmse': float(math.sqrt(sse[best] / max(n, 1)))}
    if not base:
        result['vmax_deg_s'] = None if np.isinf(vmax[best]) else float(vmax[best])
    # Flat directions of the grid are reported, not hidden: all candidates within 5 % of the best SSE.
    near = sse <= sse[best] * 1.05
    result['delay_range_s'] = [float(delays[near].min()), float(delays[near].max())]
    result['tau_range_s'] = [float(taus[near].min()), float(taus[near].max())]
    return result


def rmse(episodes, profile, channel, params):
    if not episodes:
        return None
    vmax = np.inf if params.get('vmax_deg_s') is None else params['vmax_deg_s']
    total, n = 0., 0
    for ct, cv, mt, mv in episodes:
        mt=mt[:,channel] if mt.ndim==2 else mt
        y0 = 0. if profile == 'rosmaster_dock' else mv[0, channel]
        pred = simulate(ct, cv[:, channel], mt, y0, np.array([params['delay_s']]), np.array([params['tau_s']]),
                        np.array([params.get('gain', 1.)]), np.array([vmax]))[0]
        if profile == 'rosmaster_dock':
            pred = pred + np.exp(-(mt - mt[0]) / params['tau_s']) * mv[0, channel]
        total += float(((mv[:, channel] - pred) ** 2).sum()); n += len(mt)
    return math.sqrt(total / n)


def identify(entries, profile, *, allow_synthetic=False, holdout_fraction=.3):
    """entries: [{'id','dataset'}] in recording order. Returns the candidate report."""
    if profile not in CHANNELS:
        raise ValueError('ROSMASTER profile required')
    if not math.isfinite(holdout_fraction) or not 0<holdout_fraction<1:raise ValueError('Invalid holdout fraction')
    used, data = [], []
    for entry in entries:
        if entry.get('excluded'):continue
        root = Path(entry['dataset'])
        provenance = json.loads((root / 'provenance.json').read_text())
        hardware = provenance.get('hardware', {})
        if provenance.get('synthetic') is not False and not allow_synthetic:
            raise ValueError('Only real recordings may produce a real candidate')
        if hardware.get('pipeline_check'):
            continue  # stationary diagnostic clips carry no response information
        rows = [json.loads(line) for line in (root / 'research.jsonl').read_text().splitlines()]
        s = series(rows, profile)
        if s is None:
            continue
        digest = hashlib.sha256((root / 'research.jsonl').read_bytes()).hexdigest()
        session=entry.get('session') or provenance.get('session_id')
        scene=entry.get('scene') or provenance.get('scene_id')
        if allow_synthetic:session=session or entry['id'];scene=scene or entry['id']
        if not session or not scene:raise ValueError('Нет сессии/расстановки для независимой проверки отклика')
        used.append({'id': entry['id'], 'session':session,'scene':scene,
            'research_jsonl_sha256': digest, 'synthetic': provenance.get('synthetic')}); data.append(s)
    if len(data) < 2:
        raise ValueError('Нужно минимум 2 записи с движением (одна для подгонки, одна для проверки)')
    from explorer3.datasets import connected_groups
    groups=connected_groups(used)
    if len(groups)<2:raise ValueError('Для проверки отклика нужны независимые сессии и расстановки')
    order={e['id']:i for i,e in enumerate(used)}
    groups.sort(key=lambda group:max(order[e['id']] for e in group))
    held=min(len(groups)-1,max(1,int(round(len(groups)*holdout_fraction))))
    held_ids={e['id'] for group in groups[-held:] for e in group}
    train=[d for u,d in zip(used,data) if u['id'] not in held_ids]
    test=[d for u,d in zip(used,data) if u['id'] in held_ids]
    channels = {}
    for index, name in enumerate(CHANNELS[profile]):
        if max(np.ptp(s[1][:, index]) for s in train) < 1e-6:
            channels[name] = {'status': 'not_excited', 'note': 'Канал не возбуждался в записях подгонки'}
            continue
        full = fit_channel(train, profile, index)
        simple = fit_channel(train, profile, index, simple=True)
        nominal = NOMINAL[profile]
        channels[name] = {
            'identified': full, 'simulator_form': simple,
            'holdout_rmse': {'nominal_sim': rmse(test, profile, index, nominal), 'simulator_form': rmse(test, profile, index, simple),
                             'identified': rmse(test, profile, index, full)},
            'units': ('m/s' if index < 2 else 'rad/s') if profile == 'rosmaster_dock' else 'deg'}
    return {'version': 1, 'profile': profile, 'status': 'candidate_not_deployed', 'created': time.time(),
            'model_form': __doc__.split('\n')[2:5], 'nominal': NOMINAL[profile], 'channels': channels,
            'fit_episodes': [u['id'] for u in used if u['id'] not in held_ids], 'holdout_episodes': [u['id'] for u in used if u['id'] in held_ids], 'episodes': used,
            'split_rule':'connected session/scene groups; most recent groups held out',
            'synthetic_input': any(u['synthetic'] is not False for u in used),
            'limitations': ['20 Hz frames: time constants below ~50 ms are poorly resolved',
                            'command time is the agent serial write, not MCU receipt; measurement time = snapshot minus sensor age',
                            'one fitted parameter set per channel; load, battery voltage and floor are not separated',
                            'judge only on held-out episodes; a better fit on fit episodes alone proves nothing']}


def simulator_values(candidate):
    """Values in the simulator's own parameters (single lag_s and slip for the base, single lag_s for the arm), with spread."""
    fitted = [c['simulator_form'] for c in candidate['channels'].values() if 'simulator_form' in c]
    if not fitted:
        raise ValueError('No excited channel')
    lags = [c['tau_s'] for c in fitted]
    values = {'lag_s': float(np.median(lags)), 'lag_range_s': [float(min(lags)), float(max(lags))]}
    if candidate['profile'] == 'rosmaster_dock':
        translation = [c['simulator_form']['gain'] for n, c in candidate['channels'].items() if n != 'wz_rad_s' and 'simulator_form' in c]
        values.update(slip=float(np.median(translation)) if translation else None)
    return values


def apply_to_env(env, candidate):
    """Explicit opt-in: overwrite the per-episode random lag/slip of an already reset env with identified values."""
    values = simulator_values(candidate)
    env.lag_s = values['lag_s']
    if values.get('slip') is not None and hasattr(env, 'slip'):
        env.slip = values['slip']
    return values


def synthetic_episode(root, profile, params, seed, seconds=12.):
    """Fixture generator (synthetic=True): known response -> research.jsonl in the recorder's field layout."""
    rng = np.random.default_rng(seed)
    base = profile == 'rosmaster_dock'
    k = 3 if base else 5
    t = np.arange(0., seconds, .05) + rng.normal(0, .004, int(seconds / .05))
    t.sort()
    ct = t - .01
    levels = rng.uniform(-.2, .2, (8, k)) if base else rng.uniform(60, 120, (8, k))
    cv = levels[np.minimum((ct / seconds * 8).astype(int), 7)]
    mt = t - .02
    out = np.zeros((len(mt), k))
    for c in range(k):
        p = params[c]
        vmax = np.inf if p.get('vmax_deg_s') is None else p['vmax_deg_s']
        y0 = 0. if base else 90.
        out[:, c] = simulate(ct, cv[:, c], mt, y0, np.array([p['delay_s']]), np.array([p['tau_s']]), np.array([p.get('gain', 1.)]),
                             np.array([vmax]))[0]
    out += rng.normal(0, .004 if base else .3, out.shape)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(t)):
        rows.append({'action_sent': cv[i].tolist(), 'action_sent_agent_ns': int(ct[i] * 1e9), 'measured_response': out[i].tolist(),
                     'measured_response_age_s': .02 if base else [.02] * 6, 'agent_monotonic_ns': int((t[i]) * 1e9), 'controller_ready': True})
    # the recorder pairs a frame's measurement with the NEXT snapshot time: shift by one row
    for i in range(len(rows) - 1):
        rows[i]['measured_response'] = rows[i + 1]['measured_response']
    (root / 'research.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[:-1]))
    (root / 'provenance.json').write_text(json.dumps({'synthetic': True, 'domain': 'fixture', 'hardware': {}}))
    return root
