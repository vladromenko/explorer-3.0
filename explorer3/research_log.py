"""Journal of experiment A (search: one robot vs two), one row per attempt, every outcome kept.

The program writes what it measured; the operator then confirms the result against the true place
of the object (the robot's own «found» is not the ground truth) and enters interventions.
File: DATA/research/search_trials.jsonl; CSV: /research/search_trials.csv.
"""
import csv
import io
import json
import subprocess
import time
import uuid

from .paths import DATA, ROOT

LOG = DATA / 'research' / 'search_trials.jsonl'
FIELDS = ['trial_id', 'date', 'code_version', 'scene', 'condition', 'world', 'prompt', 'robots', 'map_and_memory',
          'time_limit_s', 'program_outcome', 'program_found_by', 'seconds', 'confirmed', 'confirmation_method',
          'interventions', 'intervention_seconds', 'end_reason', 'battery_v', 'episode', 'note']


def _code_version():
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def _battery():
    out = {}
    try:
        from . import rosmaster_control
        h = rosmaster_control.request('/health', timeout=2)
        out['rosmaster'] = (h.get('observation') or {}).get('battery_voltage')
    except Exception:
        pass
    return out


def _rows():
    try:
        return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    except OSError:
        return []


def _write(rows):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOG.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tmp.replace(LOG)


def record_trial(trial, outcome, message, seconds, found, episode_id, robots):
    from . import swarm_sim
    row = {k: None for k in FIELDS}
    row.update(trial_id=uuid.uuid4().hex[:12], date=time.strftime('%Y-%m-%d %H:%M:%S'), code_version=_code_version(),
               scene=trial['scene'], condition=trial['condition'], world='sim' if swarm_sim.WORLD is not None else 'real',
               prompt=trial['prompt'], robots=','.join(robots), map_and_memory=trial.get('map_and_memory', 'as on screen'),
               time_limit_s=trial['time_limit_s'], program_outcome=outcome,
               program_found_by=found['robot'] if found else None, seconds=round(min(seconds, trial['time_limit_s']), 1),
               confirmed=None, interventions=None, end_reason=message[:200], battery_v=_battery(), episode=episode_id)
    rows = _rows()
    rows.append(row)
    _write(rows)
    return row


def confirm(trial_id, confirmed, interventions=0, intervention_seconds=0., method='оператор сверил с отметкой', note=''):
    """confirmed: 'верно' | 'ошибка' (found a wrong object/place) | 'не найдено'."""
    if confirmed not in ('верно', 'ошибка', 'не найдено'):
        raise ValueError('Выберите: верно / ошибка / не найдено')
    rows = _rows()
    for r in rows:
        if r['trial_id'] == trial_id:
            r.update(confirmed=confirmed, interventions=int(interventions), intervention_seconds=float(intervention_seconds),
                     confirmation_method=method, note=note[:300])
            _write(rows)
            return r
    raise ValueError('Попытка не найдена')


def rows():
    return _rows()


def to_csv():
    buf = io.StringIO()
    w = csv.DictWriter(buf, FIELDS)
    w.writeheader()
    for r in _rows():
        w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v) for k, v in r.items() if k in FIELDS})
    return buf.getvalue()
