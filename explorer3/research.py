"""Small intervention study module; never drives robots or infers counterfactual labels."""
import hashlib
import json
import math
import secrets
import time
import uuid
from pathlib import Path
import numpy as np
from . import catalog
from .paths import DATA
from .profiles import profile

FEATURES = ('command_change', 'response_change', 'tracking_error', 'observation_age_s',
            'camera_age_s', 'send_latency_s', 'progress', 'assigned_delay_s')
PHYSICAL = {'response_change', 'tracking_error'}


def features(history, decision_ns, scales, delay_s=0., progress=None):
    """Only values received before decision time; no outcome or robot ID features."""
    rows = [r for r in history if decision_ns-3_000_000_000 <= r['observation_timestamp_ns'] <= decision_ns
            and r['action_send_end_ns'] <= decision_ns]
    if not rows:
        raise ValueError('No past observations at the decision time')
    scale = np.asarray(scales, dtype=float)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Positive channel scales required')
    actions = np.asarray([r['action'] for r in rows], dtype=float)
    if actions.shape[1:] != scale.shape or not np.isfinite(actions).all():
        raise ValueError('Action channels do not match physical scales')
    if any(not r['observation_timestamp_ns'] <= r['action_send_start_ns'] <= r['action_send_end_ns'] for r in rows):
        raise ValueError('Noncausal telemetry')
    dt=np.diff([r['action_send_end_ns'] for r in rows])/1e9
    if (dt<=0).any():raise ValueError('Nonmonotonic command history')
    last = rows[-1]
    measured = [r for r in rows if r.get('state_valid') is True
                and r.get('measured_response_kind') == 'measured_joint_position']
    response, residual = None, None
    if len(measured) >= 2 and measured[-1] is last:
        states = np.asarray([r['observation.state'] for r in measured], dtype=float)
        if states.shape[1:] == scale.shape and np.isfinite(states).all():
            state_dt=np.diff([r['observation_timestamp_ns'] for r in measured])/1e9
            if (state_dt<=0).any():raise ValueError('Nonmonotonic measured history')
            response = float(np.abs(np.diff(states, axis=0)/scale/state_dt[:,None]).mean())
            # Compare current measured state with a command sent BEFORE it was observed.
            prior = [r for r in rows if r['action_send_end_ns'] < last['observation_timestamp_ns']]
            if prior:
                residual = float(np.abs((states[-1]-np.asarray(prior[-1]['action']))/scale).mean())
    camera_ns = last.get('camera', {}).get('timestamp_ns')
    camera_age = (decision_ns-camera_ns)/1e9 if camera_ns is not None and camera_ns <= decision_ns else None
    if progress is not None and (not math.isfinite(progress) or not 0 <= progress <= 1):
        raise ValueError('Online progress must be bounded; never use final outcome')
    return dict(zip(FEATURES, [float(np.abs(np.diff(actions, axis=0)/scale/dt[:,None]).mean()) if len(actions)>1 else None,
        response, residual, (decision_ns-last['observation_timestamp_ns'])/1e9, camera_age,
        (last['action_send_end_ns']-last['action_send_start_ns'])/1e9, progress, float(delay_s)]))


def initialize():
    with catalog.connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS studies(id TEXT PRIMARY KEY, config TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions(
          episode_id TEXT PRIMARY KEY, study_id TEXT NOT NULL, record TEXT NOT NULL, created REAL NOT NULL);
        ''')


def create_study(name, policy_checksum, delays=(0.,), correction_limit=10.):
    profile(name)
    if len(policy_checksum) != 64 or any(c not in '0123456789abcdef' for c in policy_checksum):
        raise ValueError('A frozen policy SHA256 is required')
    if not delays or any(not math.isfinite(d) or not 0 <= d <= 30 for d in delays):
        raise ValueError('Delay schedule invalid')
    if not math.isfinite(correction_limit) or not 0 < correction_limit <= 10:
        raise ValueError('Correction limit must be in (0,10] seconds')
    ident = uuid.uuid4().hex
    cfg = {'profile': name, 'policy_sha256': policy_checksum, 'assigned_delays_s': list(delays),
           'correction_limit_s': correction_limit, 'probability': .5, 'waiting_behavior': 'hold',
           'status': 'draft_pending_physical_timing_acceptance', 'assignment_unit': 'one_per_episode'}
    with catalog.connect() as db:
        db.execute('INSERT INTO studies VALUES(?,?,?)', (ident, json.dumps(cfg), time.time()))
    return ident


def studies():
    with catalog.connect() as db:
        return [{'id': r['id'], **json.loads(r['config'])} for r in db.execute('SELECT * FROM studies ORDER BY created DESC')]


def assign(study_id, episode_id, session_id, scene_id, decision_ns, online_features, timing_accepted=False):
    study = next((s for s in studies() if s['id'] == study_id), None)
    if study is None:
        raise ValueError('Unknown study')
    if not timing_accepted:
        raise ValueError('Physical timing and safe decision point must be accepted first')
    if set(online_features) != set(FEATURES):
        raise ValueError('Online feature schema mismatch')
    if set(online_features) & {'outcome', 'success', 'robot_id'}:
        raise ValueError('Outcome and identity are not online features')
    delay = secrets.choice(study['assigned_delays_s'])
    record = {'episode_id': episode_id, 'study_id': study_id, 'session': session_id, 'scene': scene_id,
              'profile': study['profile'], 'policy_sha256': study['policy_sha256'], 'decision_time_ns': decision_ns,
              'assigned': secrets.randbelow(2), 'propensity': .5, 'assigned_delay_s': delay,
              'features': {**online_features, 'assigned_delay_s': delay}, 'outcome': None,
              'emergency': False, 'deviation': None, 'events': [], 'synthetic': False,'deadline_ns':decision_ns+int((delay+study['correction_limit_s'])*1e9),
              'clock_domain':'homebrain_monotonic_ns'}
    with catalog.connect() as db:
        db.execute('INSERT INTO decisions VALUES(?,?,?,?)', (episode_id, study_id, json.dumps(record), time.time()))
    return record


def record_event(episode_id, kind, monotonic_ns, **values):
    allowed = {'request', 'human_start', 'human_end', 'switch_start', 'switch_end', 'reset_start', 'reset_end', 'finish','request_sent','request_received','queue_enter','queue_leave','operator_notified','operator_response'}
    if kind not in allowed:
        raise ValueError('Unknown study event')
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT record FROM decisions WHERE episode_id=?', (episode_id,)).fetchone()
        if row is None:
            raise ValueError('No assignment for this episode')
        record = json.loads(row['record'])
        latest = record['events'][-1]['time_ns'] if record['events'] else record['decision_time_ns']
        existing={e['kind'] for e in record['events']}
        pairs={'human_end':'human_start','switch_end':'switch_start','reset_end':'reset_start'}
        if kind in existing or (kind in pairs and pairs[kind] not in existing):
            raise ValueError('Repeated event or end without start')
        if monotonic_ns < latest or (record['outcome'] is not None and kind not in {'reset_start','reset_end'}):
            raise ValueError('Nonmonotonic event or episode already finished')
        if kind == 'finish':
            if values.get('outcome') not in {'success', 'failure', 'aborted'}:
                raise ValueError('An explicit episode outcome is required')
            record.update(outcome=values['outcome'], emergency=bool(values.get('emergency')),
                          deviation=values.get('deviation'))
        record['events'].append({'kind': kind, 'time_ns': monotonic_ns, **values})
        db.execute('UPDATE decisions SET record=? WHERE episode_id=?', (json.dumps(record), episode_id))


def records():
    with catalog.connect() as db:
        return [json.loads(r['record']) for r in db.execute('SELECT record FROM decisions ORDER BY created')]


def matrix(rows, include_physical=True, common_sensors=False):
    names = [n for n in FEATURES if (include_physical and not common_sensors) or n not in PHYSICAL]
    # An explicit mask for EVERY feature avoids changing dimensionality at transfer time.
    data = []
    for row in rows:
        values, masks = [], []
        for name in names:
            value = row['features'].get(name)
            valid = value is not None and math.isfinite(value)
            values.append(float(value) if valid else 0.)
            masks.append(float(valid))
        data.append(values + masks)
    return np.asarray(data), names + [n+'_available' for n in names]


def fit(rows, output, include_physical=True, source_only=True, synthetic=False, minimum_per_arm=20):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    selected = [r for r in rows if r['outcome'] in {'success', 'failure'}]
    if not selected or any(bool(r.get('synthetic')) != synthetic for r in selected):
        raise ValueError('Do not mix synthetic fixtures with physical data')
    if source_only and any(r['profile'].startswith('rosmaster') for r in selected):
        raise ValueError('Both ROSMASTER profiles are excluded from source training')
    if len({r['episode_id'] for r in selected}) != len(selected):
        raise ValueError('Only one decision per episode is allowed')
    x, names = matrix(selected, include_physical)
    models = []
    for arm in (0, 1):
        indices = [i for i, r in enumerate(selected) if r['assigned'] == arm]
        y = [int(selected[i]['outcome'] == 'success') for i in indices]
        if len(indices) < (4 if synthetic else minimum_per_arm) or len(set(y)) != 2:
            raise ValueError('Each assignment arm needs both outcomes and ≥20 episodes (pilot only)')
        if any(not 0 < selected[i]['propensity'] < 1 for i in indices):
            raise ValueError('Missing randomized assignment propensity')
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=1000))
        weights = [1/(selected[i]['propensity'] if arm else 1-selected[i]['propensity']) for i in indices]
        model.fit(x[indices], y, logisticregression__sample_weight=weights)
        models.append(model)
    artifact = {'models': models, 'feature_names': names, 'include_physical': include_physical,
                'training_episodes': [r['episode_id'] for r in selected],
                'training_groups': [(r['session'], r['scene']) for r in selected],
                'profiles': sorted({r['profile'] for r in selected}), 'synthetic': synthetic,
                'estimand': 'assigned_correction_intention_to_treat', 'normalization': 'train_only',
                'source_only': source_only, 'missing_or_aborted_outcomes': len(rows)-len(selected), 'emergency_episodes':sum(bool(r.get('emergency')) for r in selected)}
    joblib.dump(artifact, output)
    return artifact


def predict(artifact, rows):
    x, names = matrix(rows, artifact['include_physical'])
    if names != artifact['feature_names']:
        raise ValueError('Feature contract mismatch')
    p0, p1 = [model.predict_proba(x)[:, 1] for model in artifact['models']]
    return [{'p_without': float(a), 'p_with': float(b), 'gain': float(b-a)} for a,b in zip(p0,p1)]


def prioritize(requests, method='gain', seed=1000):
    """Pure selection; does not start a robot or a correction."""
    import random
    pending = [dict(r) for r in requests if not r.get('expired')]
    if method == 'random':
        random.Random(seed).shuffle(pending)
    elif method == 'fifo':
        pending.sort(key=lambda r: r['request_time_ns'])
    elif method in {'risk', 'gain'}:
        if method=='gain':pending=[r for r in pending if math.isfinite(r['gain']) and r['gain']>0]
        for r in pending:
            cost = r['switch_seconds'] + r['correction_seconds']
            if cost <= 0 or not math.isfinite(cost):
                raise ValueError('Expected operator time must be positive')
            r['priority'] = max(0., r['gain'] if method == 'gain' else 1-r['p_without']) / cost
        pending.sort(key=lambda r: (-r['priority'], r['request_time_ns']))
    else:
        raise ValueError('Unknown scheduler baseline')
    return pending


def report(artifact, rows, seed=1000):
    selected = [r for r in rows if r['outcome'] in {'success','failure'}]
    train_sessions = {g[0] for g in artifact['training_groups']}
    train_scenes = {g[1] for g in artifact['training_groups']}
    if any(r['episode_id'] in artifact['training_episodes'] or r['session'] in train_sessions
           or r['scene'] in train_scenes for r in selected):
        raise ValueError('Evaluation overlaps a training episode, session or scene')
    if not selected:
        raise ValueError('No held-out completed episodes')
    predictions = predict(artifact, selected)
    per_profile = {}
    for name in sorted({r['profile'] for r in selected}):
        subset = [(r,p) for r,p in zip(selected,predictions) if r['profile']==name]
        from .datasets import connected_groups
        components=connected_groups([dict(r,id=r['episode_id']) for r,p in subset])
        groups={str(i):[(r,p) for r,p in subset if r['episode_id'] in {e['id'] for e in group}]
                for i,group in enumerate(components)}
        def metrics(items):
            return {'success_rate': float(np.mean([r['outcome']=='success' for r,p in items])),
                    'brier_assigned': float(np.mean([(p['p_with' if r['assigned'] else 'p_without']-
                                                   int(r['outcome']=='success'))**2 for r,p in items]))}
        point = metrics(subset)
        rng = np.random.default_rng(seed)
        keys = list(groups)
        samples = [metrics([item for key in rng.choice(keys, len(keys), replace=True) for item in groups[key]])
                   for _ in range(500)] if len(keys)>1 else []
        ci = {k: np.percentile([m[k] for m in samples], [2.5,97.5]).tolist() for k in point} if samples else None
        durations = {'human_seconds': None, 'switch_seconds': None, 'reset_seconds': None, 'decision_to_end_seconds': None, 'request_to_human_seconds': None, 'queue_seconds':None, 'reaction_seconds':None, 'request_roundtrip_seconds':None}
        timing_counts = {k:0 for k in durations}
        for r,_ in subset:
            events = {e['kind']: e['time_ns'] for e in r['events']}
            for key, begin, end in [('human_seconds','human_start','human_end'), ('switch_seconds','switch_start','switch_end'),
                                    ('reset_seconds','reset_start','reset_end'), ('queue_seconds','queue_enter','queue_leave'),
                                    ('reaction_seconds','operator_notified','operator_response'),
                                    ('request_roundtrip_seconds','request_sent','request_received')]:
                if begin in events and end in events:
                    durations[key] = (durations[key] or 0.) + (events[end]-events[begin])/1e9
                    timing_counts[key] += 1
            if 'finish' in events:
                durations['decision_to_end_seconds'] = (durations['decision_to_end_seconds'] or 0.) + (events['finish']-r['decision_time_ns'])/1e9
                timing_counts['decision_to_end_seconds'] += 1
            if 'request' in events and 'human_start' in events:
                durations['request_to_human_seconds'] = (durations['request_to_human_seconds'] or 0.) + (events['human_start']-events['request'])/1e9
                timing_counts['request_to_human_seconds'] += 1
        calibration=[]
        for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            bucket=[(r,p) for r,p in subset if low<=p['p_with' if r['assigned'] else 'p_without']<high
                    or (high==1 and p['p_with' if r['assigned'] else 'p_without']==1)]
            calibration.append({'range':[float(low),float(high)],'episodes':len(bucket),
                'mean_prediction':float(np.mean([p['p_with' if r['assigned'] else 'p_without'] for r,p in bucket])) if bucket else None,
                'observed_success':float(np.mean([r['outcome']=='success' for r,p in bucket])) if bucket else None})
        per_profile[name] = {'calibration':calibration,**point, 'episodes': len(subset), 'independent_session_scene_groups': len(keys), 'group_bootstrap_ci95': ci,
                             'time': durations, 'timing_observed_episodes':timing_counts, 'deviations': sum(bool(r.get('deviation')) for r,p in subset),
                             'emergencies':sum(bool(r.get('emergency')) for r,p in subset)}
    return {'profiles': per_profile,
            'macro_success_rate': float(np.mean([v['success_rate'] for v in per_profile.values()])),
            'synthetic': artifact['synthetic'], 'causal_effect_proven': False,
            'missing_or_aborted_outcomes': len(rows)-len(selected), 'emergency_episodes':sum(bool(r.get('emergency')) for r in selected),
            'limitation': 'Intention-to-treat prediction; no observed individual counterfactual. Missing timing is not zero measured cost. CI unavailable for one independent session/scene group.'}


def analyze(output):
    """Freeze a local decision snapshot and compare transfer budgets without control actions."""
    from .datasets import grouped_split
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records() if r['outcome'] in {'success','failure'}]
    if not rows:
        raise ValueError('Нет завершённых рандомизированных эпизодов; показы не являются данными эффекта помощи')
    snapshot = json.dumps(rows, sort_keys=True, indent=2)
    (output/'decisions.json').write_text(snapshot)
    split = grouped_split([dict(r,id=r['episode_id']) for r in rows])
    (output/'split.json').write_text(json.dumps({k:[r['episode_id'] for r in v] for k,v in split.items()},indent=2))
    source = [r for r in split['train'] if r['profile'] in {'so101_pick_place','tank_dock'}]
    target = sorted([r for r in split['train'] if r['profile'].startswith('rosmaster')],key=lambda r:r['episode_id'])
    held_out = [r for r in split['test'] if r['profile'].startswith('rosmaster')]
    if {r['profile'] for r in source} != {'so101_pick_place','tank_dock'} or not held_out:
        raise ValueError('Нужны обе исходные платформы и отдельные тестовые эпизоды ROSMASTER')
    results = {}
    for budget in (0,10,20):
        for physical in (False,True):
            for transfer in (True,False):
                if transfer or budget:
                    name=f"{'transfer' if transfer else 'target-only'}-{budget}-{'physical' if physical else 'commands'}"
                    if len(target)<budget:
                        results[name]={'status':'insufficient_target_budget'}
                    else:
                        train=(source if transfer else [])+target[:budget]
                        try:
                            model=fit(train,output/(name+'.joblib'),include_physical=physical,
                                      source_only=budget==0,minimum_per_arm=20 if budget==0 else 2)
                            results[name]={'status':'completed','source_episodes':len(source) if transfer else 0,
                                           'target_episodes':budget,'report':report(model,held_out),
                                           'small_target_budget':budget in {10,20}}
                        except ValueError as exc:
                            results[name]={'status':'insufficient_data','reason':str(exc)}
    result={'snapshot_sha256':hashlib.sha256(snapshot.encode()).hexdigest(),'variants':results,
            'physical_skill_verified':False,'causal_effect_proven':False,
            'common_sensor_baseline':'commands variant masks out all measured response features',
            'scheduler_experiment':'not performed; random/fifo/risk/gain selectors prepared, policies not executed'}
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result
