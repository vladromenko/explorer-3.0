"""Real-robot pipeline check for ROSMASTER without motion, in an isolated data directory.

Records short episodes from the live agent snapshot while NO control session is armed (nothing
is sent to the robot), then validates them, runs the official ACT smoke training (50 steps),
reloads the checkpoint in a new process and runs inference without sending commands. The
episodes are not demonstrations and never enter the research catalogue.
Usage: EXPLORER3_DATA=<isolated dir> python scripts/rosmaster_pipeline_check.py
"""
import json, os, sys, time
from explorer3.paths import DATA, prepare
from explorer3 import catalog
from explorer3.rosmaster_session import RosmasterSession, PROFILES
from explorer3.jobs import Jobs

assert 'pipeline-check' in str(DATA), 'Run only in an isolated pipeline-check data directory'
prepare(); catalog.initialize()


class Idle:
    owner = None; lease = None; status = {}; error = None; maintenance = False
    def set_direct(self, value): pass


report = {'started': time.time(), 'motion_commands_sent': 'none (no armed session)', 'profiles': {}}
session = RosmasterSession(Idle())
for name in PROFILES:
    episodes = []
    for index in range(3):
        session.start('pipeline-check', name, f'check-{index}', check=True, collection=f'check-{name}-{index}')
        end = time.monotonic() + 60
        while time.monotonic() < end and session.status()['frames'] < 60:
            time.sleep(.1)
        session.finish('success'); session.thread.join(60)
        status = session.status()
        if status['error']:
            raise SystemExit(f'{name}: {status["error"]}')
        catalog.exclude(status['episode'], False)  # isolated catalogue only: allow the smoke trainer to read it
        from explorer3.validation import validate
        result = validate(status['episode'])
        episodes.append({k: result[k] for k in ('episode_id', 'frames', 'actual_hz', 'interval_ms_p50_p95_max',
                                                 'reused_camera_frames', 'camera_age_ms_p95', 'readback')})
    report['profiles'][name] = {'episodes': episodes}
    print(name, json.dumps(episodes), flush=True)

jobs = Jobs(Idle())
for name in PROFILES:
    ident = jobs.enqueue(name, 'act', smoke=True)
    jobs.tick()
    while jobs.thread and jobs.thread.is_alive():
        time.sleep(2)
    job = jobs.get(ident)
    work = DATA/'jobs'/ident
    result = json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else None
    reload = None
    if result:
        reload_path = __import__('pathlib').Path(result['checkpoint']).parent/'explorer3-reload.json'
        reload = json.loads(reload_path.read_text()) if reload_path.exists() else None
    report['profiles'][name]['smoke'] = {'job': ident, 'status': job['status'], 'error': job['error'], 'result': result, 'reload': reload}
    print(name, job['status'], json.dumps(reload), flush=True)
report['finished'] = time.time()
(DATA/'pipeline-check-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print('REPORT', DATA/'pipeline-check-report.json')
