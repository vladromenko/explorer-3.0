"""Run a ROSMASTER recording against an in-process fake agent; prints the episode validation (synthetic)."""
import json, sys, time, threading
import cv2, httpx, numpy as np
from explorer3 import rosmaster_session as rs, catalog
from explorer3.paths import DATA, prepare

NAME = sys.argv[1]
PILOT = len(sys.argv) > 2 and sys.argv[2] == 'pilot'
start = time.monotonic_ns()
step = [0]
lock = threading.Lock()


def handler(request):
    now = time.monotonic_ns() - 5_000_000_000  # robot clock differs from Homebrain
    if request.url.path == '/health':
        return httpx.Response(200, json={'monotonic_ns': now})
    with lock:
        step[0] += 1; k = step[0]
    image = np.full((480, 640, 3), (k * 7) % 255, np.uint8)
    ok, jpeg = cv2.imencode('.jpg', image)
    t = k * .05
    source = 'human' if (not PILOT or OP.source == 'human') else 'policy'
    state = {'agent_monotonic_ns': now, 'control_step': k, 'armed': True, 'profile': NAME, 'preparing': False,
             'mode': 'manual', 'command_source': source, 'operator_intent': {'drive': [.5, 0, 0], 'arm': [0, 0, 1]},
             'direct_requested': None, 'direct_received_ns': None,
             'base_requested': [.15, 0, 0], 'base_after_limits': [.1, 0, 0], 'base_guard': None,
             'base_sent': [.1 * min(1, t), 0., 0.], 'base_sent_ns': now - 1_000_000,
             'arm_sent': [90, 90 + t, 90, 90, 90, 60], 'arm_sent_ns': now - 1_000_000, 'arm_blocked': None,
             'arm_target': [90, 90 + t, 90, 90, 90, 60], 'arm_measured': [90, 89.5 + t, 90, 90, 90, 60],
             'arm_measured_ages_s': [.01] * 6, 'arm_measured_valid': [True] * 6,
             'base_measured': [.09 * min(1, t), 0., 0.], 'base_measured_age_s': .02,
             'battery_voltage': 11.7, 'battery_age_s': .1, 'controller_ready': True, 'arm_cancel_recoveries': 0,
             'motion_commands_sent': k, 'clock_resyncs': 1}
    camera = {'available': True, 'shape': [480, 640, 3], 'received_ns': now - 20_000_000,
              'source_timestamp_ns': 123, 'source_clock': 'ros_header_time_not_homebrain', 'sequence': k}
    return httpx.Response(200, content=jpeg.tobytes(), headers={'X-Explorer-Sample': json.dumps(state),
                                                                'X-Explorer-Camera': json.dumps(camera)})


class Operator:
    owner = None; lease = None; status = {}; error = None; source = None
    def set_direct(self, value): pass
    def set_source(self, value): self.source = value


OP = Operator()


prepare(); catalog.initialize()
ENVIRONMENT=NAME.endswith('_rgbd_lidar')
if ENVIRONMENT:
    from explorer3 import sensors,profiles
    from explorer3.recording import EpisodeWriter
    from rosmaster_sensor_fixture import FixtureFeed
    sensors.SensorFeed=FixtureFeed
    profiles.PROFILES[NAME]['domain']='sim'  # fixture-only schema, confined to this child process
    rs.EpisodeWriter=lambda *a,**kw:EpisodeWriter(*a,**kw,domain='fixture')
rs.agent_client = lambda timeout=.4: httpx.Client(transport=httpx.MockTransport(handler), base_url='http://agent')
session = rs.RosmasterSession(OP)
if PILOT:
    from explorer3 import trials, research
    class Policy:
        decision_info = {}
        def __init__(self, plan): pass
        def next(self, observation):
            if ENVIRONMENT:assert len(observation['observation.environment_state'])==265
            return [0.]*3 if rs.control_profile(NAME)=='rosmaster_dock' else list(observation['observation.state'])
        def reset(self): pass
    trials.PolicyActions = Policy
    research.initialize()
    research.secrets.randbelow = lambda n: 1          # force the correction branch for the test
    study_id = research.create_study(NAME, 'a' * 64, delays=(.3,), correction_limit=1.)
    research.accept_timing(study_id, 1., [.3])
    study = next(s for s in research.studies() if s['id'] == study_id)
    plan = {'id': 'p' * 32, 'kind': 'policy', 'profile': NAME, 'checkpoint_sha256': 'a' * 64, 'max_duration_s': 30.}
    session.start('test-owner', NAME, 'scene-test', check=True, plan=plan, study=study)
else:
    session.start('test-owner', NAME, 'scene-test', collection='r3-series-01', check=True)
end = time.monotonic() + 60
while time.monotonic() < end and session.status()['frames'] < (70 if PILOT else 30):
    time.sleep(.1)
session.finish('success'); session.thread.join(30)
status = session.status()
assert status['error'] is None, status
from explorer3.validation import validate
rows = catalog.rows('episodes')
assert all(r['excluded'] == 1 for r in rows)
if not PILOT:assert all(r['session']=='r3-series-01' and r['scene']=='scene-test' for r in rows)
result = validate(rows[0]['id'],domain='fixture' if ENVIRONMENT else 'real')
extra = {}
if PILOT:
    from explorer3 import research
    record = research.records()[0]
    extra = {'events': [e['kind'] for e in record['events']], 'assigned': record['assigned'], 'outcome': record['outcome'],
             'categories': [r['category'] for r in sorted(rows, key=lambda r: r['created'])], 'features': record['features']}
print(json.dumps({'status': status, 'frames': result['frames'], 'hz': result['actual_hz'], **extra}))
