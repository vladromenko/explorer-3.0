"""ROSMASTER demonstrations, replays and policy trials recorded as LeRobot episodes.

The operator drives in the control panel; this module only samples the agent at the profile
rate. Each sample is taken atomically on the robot (camera frame first, then the control
state), so observation k pairs with the command in force at the next sample k+1, which was
decided after observation k. Robot times are mapped to the Homebrain clock with a measured
midpoint offset; they are never treated as a shared clock.
"""
import json
import threading
import time
import cv2
import httpx
import numpy as np
from . import catalog
from .paths import DATA
from .profiles import profile
from .recording import EpisodeWriter
from .resources import ResourceLease

PROFILES = ('rosmaster_dock', 'rosmaster_pick_place')
BASE_FEEDBACK_MAX_AGE_S = .5
ARM_FEEDBACK_MAX_AGE_S = .6


def agent_client(timeout=.4):
    cfg = json.loads((DATA/'configs/rosmaster-agent.json').read_text())
    return httpx.Client(base_url=f"http://{cfg['host']}:{cfg['port']}", headers={'X-Explorer-Token': cfg['token']},
                        timeout=timeout, trust_env=False)


class ClockMap:
    """Agent monotonic ns -> Homebrain monotonic ns via the lowest-RTT midpoint of recent probes."""

    def __init__(self, http):
        self.http = http
        self.offset_ns = None
        self.rtt_ns = None
        self.probes = []

    def probe(self, count=5):
        for _ in range(count):
            sent = time.monotonic_ns()
            response = self.http.get('/health')
            received = time.monotonic_ns()
            response.raise_for_status()
            agent = response.json()['monotonic_ns']
            self.probes = (self.probes + [(received - sent, (sent + received)//2 - agent)])[-20:]
        self.rtt_ns, self.offset_ns = min(self.probes)
        return {'offset_ns': self.offset_ns, 'rtt_ns': self.rtt_ns, 'kind': 'midpoint_estimate_not_shared_clock'}

    def to_homebrain(self, agent_ns):
        return None if agent_ns is None else int(agent_ns + self.offset_ns)


def state_and_action(name, current, following):
    """observation.state from sample k and the action in force at sample k+1 (no future data in state)."""
    if name == 'rosmaster_dock':
        measured = current['base_measured']
        age = current['base_measured_age_s']
        if measured is None or age is None or age > BASE_FEEDBACK_MAX_AGE_S:
            raise RuntimeError('Нет свежей измеренной скорости базы; эпизод прерван')
        return [*map(float, measured), *map(float, current['base_sent'])], [float(v) for v in following['base_sent']]
    if current['arm_measured'] is None:
        raise RuntimeError('Нет свежих измерений всех шести суставов; эпизод прерван')
    return [float(v) for v in current['arm_measured']], [float(v) for v in following['arm_target']]


class RosmasterSession:
    """One recording/trial at a time; the trainer cannot run concurrently (compute lease)."""

    def __init__(self, operator):
        self.operator = operator
        self.lock = threading.RLock()
        self.thread = None
        self.stop_event = threading.Event()
        self.outcome = 'aborted'
        self.state = {'phase': 'idle', 'frames': 0, 'error': None, 'profile': None, 'episode': None, 'hz': None}
        self.provider = None
        self.plan = None

    def status(self):
        with self.lock:
            return dict(self.state)

    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, owner, name, scene, category='demonstration', collection=None, check=False, plan=None, study=None):
        """check=True records without an armed control session (pipeline check, nothing is sent).
        plan: replay/policy trial; study: diagnostic pilot (one random correction assignment per trial)."""
        if name not in PROFILES:
            raise ValueError('Профиль ROSMASTER: rosmaster_dock или rosmaster_pick_place')
        if self.active():
            raise RuntimeError('Запись уже идёт')
        if not scene.strip():
            raise ValueError('Нужно обозначение расстановки')
        if category not in {'demonstration', 'correction'}:
            raise ValueError('Неизвестный тип записи')
        if study is not None:
            if plan is None or plan['kind'] != 'policy' or study['profile'] != name:
                raise ValueError('Пилот проводится только с испытанием политики этой задачи')
            if study.get('policy_sha256') != plan.get('checksum_sha256', plan.get('checkpoint_sha256')):
                raise ValueError('Пилот привязан к другой замороженной политике')
            if not study.get('timing_accepted'):
                raise PermissionError('Сначала подтвердите безопасную точку решения и задержки (страница исследования)')
        if not check:
            if self.operator.owner != owner or not self.operator.lease:
                raise PermissionError('Сначала «Включить управление» для этой задачи')
            if (self.operator.status or {}).get('profile') != name:
                raise PermissionError('Управление включено не для этой задачи')
        self.stop_event.clear()
        self.outcome = 'aborted'
        self.emergency = False
        self.correction_done = threading.Event()
        self.plan = plan
        self.study = study
        with self.lock:
            self.state = {'phase': 'preparing', 'frames': 0, 'error': None, 'profile': name, 'episode': None,
                          'hz': None, 'category': category, 'scene': scene, 'pilot': None}
        self.thread = threading.Thread(target=self.run, args=(owner, name, scene, category, collection, check),
                                       daemon=True)
        self.thread.start()

    def finish(self, outcome, emergency=False):
        if outcome not in {'success', 'failure', 'aborted'}:
            raise ValueError('Исход: success, failure или aborted')
        self.outcome = outcome
        self.emergency = self.emergency or bool(emergency)
        self.stop_event.set()

    def end_correction(self):
        self.correction_done.set()

    def pilot(self, **values):
        with self.lock:
            self.state['pilot'] = {**(self.state.get('pilot') or {}), **values}

    def run(self, owner, name, scene, category, collection, check):
        from . import research
        cfg = profile(name)
        period = 1/cfg['fps']
        lease = ResourceLease('compute')
        http = agent_client()
        provider = None
        pool = []
        segments = []           # one LeRobot episode per uninterrupted command source
        history = []            # recorded frames for online pilot features (past only)
        trial_id = None
        decision = None
        session_id = collection or f'{time.strftime("%Y%m%d")}-{owner[:8]}'
        provenance = {}
        try:
            lease.__enter__()
            clock = ClockMap(http)
            provenance = {'robot': 'rosmaster', 'profile_version': cfg['version'], 'clock_mapping': clock.probe(),
                          'agent': 'explorer3 native serial agent', 'pipeline_check': check,
                          'motor_acknowledgement': False, 'physical_skill_verified': False}
            if self.plan:
                from .trials import ReplayActions, PolicyActions
                provider = (ReplayActions if self.plan['kind'] == 'replay' else PolicyActions)(self.plan)
                provenance['trial_plan'] = self.plan
                trial_id = __import__('uuid').uuid4().hex
                provenance['trial_id'] = trial_id
            # LeRobotDataset.create takes seconds: create every segment the run can need before sampling.
            plan_sources = (['policy', 'human', 'policy'] if self.study else ['policy']) if self.plan else ['human']
            pool = [{'writer': EpisodeWriter(name, session_id, scene, category=(
                        ('rollout' if source == 'policy' else 'correction') if self.plan else category)),
                     'source': source, 'frames': 0} for source in plan_sources]
            with self.lock:
                self.state.update(phase='recording')
            previous = None
            deadline = time.monotonic()
            started = deadline
            phase = 'policy' if provider else 'human'
            hold_until = human_until = None
            human_started = False
            while not self.stop_event.is_set():
                if not check and not self.operator.lease:
                    raise InterruptedError(self.operator.error or 'Управление остановлено')
                now = time.monotonic()
                if self.plan and now-started > self.plan['max_duration_s']:
                    raise InterruptedError('Лимит времени испытания')
                sample = self.sample(http, clock)
                if self.study and decision is None and now-started >= self.study['decision_after_s'] and history:
                    decision_ns = time.monotonic_ns()
                    online = research.features(history, decision_ns, cfg['feature_scales'])
                    decision = research.assign(self.study['id'], trial_id, session_id, scene, decision_ns,
                                               {k: v for k, v in online.items() if k != 'assigned_delay_s'} | {'assigned_delay_s': 0.},
                                               timing_accepted=True)
                    provenance['pilot_decision'] = decision
                    self.pilot(assigned=bool(decision['assigned']), delay_s=decision['assigned_delay_s'], phase='decided')
                    if decision['assigned']:
                        research.record_event(trial_id, 'request', decision_ns)
                        phase, hold_until = 'hold', now+decision['assigned_delay_s']
                        self.operator.set_direct(None)
                if phase == 'hold' and now >= hold_until:
                    research.record_event(trial_id, 'operator_notified', time.monotonic_ns())
                    self.operator.set_source('human')
                    self.correction_done.clear()
                    phase, human_until = 'human', now+self.study['correction_limit_s']
                    self.pilot(phase='human', until=time.time()+self.study['correction_limit_s'])
                if phase == 'human' and self.plan:
                    active = any(abs(v) > 1e-3 for key in ('drive', 'arm') for v in sample['operator_intent'].get(key, [])) or any(
                        abs(sample['operator_intent'].get(k, 0)) > 1e-3 for k in ('pitch', 'roll', 'grip'))
                    if active and not human_started:
                        research.record_event(trial_id, 'human_start', time.monotonic_ns())
                        human_started = True
                    if now >= human_until or self.correction_done.is_set():
                        if not human_started:
                            research.record_event(trial_id, 'human_start', time.monotonic_ns(), performed=False)
                        research.record_event(trial_id, 'human_end', time.monotonic_ns())
                        self.operator.set_source('policy')
                        provider.reset()  # stale action chunks from before the correction are dropped
                        phase = 'policy'
                        self.pilot(phase='policy_after_correction')
                if provider is not None and phase == 'policy':
                    observation = {'observation.state': state_and_action(name, sample, sample)[0],
                                   'images': {cfg['camera']: {'image': sample['rgb']}},
                                   'observation_timestamp_ns': sample['observation_ns']}
                    action = provider.next(observation)
                    self.operator.set_direct({'base': action} if name == 'rosmaster_dock' else {'arm': action})
                    sample['requested_action'] = list(action)
                    sample['decision'] = dict(getattr(provider, 'decision_info', {}))
                if previous is not None:
                    frame = self.frame(name, cfg, previous, sample, 0)
                    source = frame['command_source']
                    if not segments or segments[-1]['source'] != source:
                        if not pool or pool[0]['source'] != source:
                            raise RuntimeError('Неожиданная смена источника команд: '+source)
                        segments.append(pool.pop(0))
                        with self.lock:
                            self.state['episode'] = segments[-1]['writer'].id
                    segment = segments[-1]
                    frame['frame_id'] = segment['frames']
                    frame['trial_id'] = trial_id
                    frame['segment_index'] = len(segments)-1
                    segment['writer'].add(frame)
                    segment['frames'] += 1
                    history.append({**{k: frame[k] for k in ('observation_timestamp_ns', 'action_send_start_ns', 'action_send_end_ns',
                                                             'action', 'measured_response_kind', 'camera_ns_mapped')
                                       if k in frame},
                                    'observation.state': previous['base_measured'] if name == 'rosmaster_dock' else frame['observation.state'],
                                    'state_valid': True, 'camera': {'timestamp_ns': previous['camera_ns']}})
                    history[:] = history[-120:]
                    with self.lock:
                        self.state['frames'] += 1
                        elapsed = time.monotonic()-started
                        self.state['hz'] = round(self.state['frames']/elapsed, 1) if elapsed > 0 else None
                previous = sample
                deadline += period
                lag = time.monotonic()-deadline
                if lag > .5:
                    raise RuntimeError('Запись не выдерживает частоту профиля')
                if lag > 0:
                    deadline = time.monotonic()
                self.stop_event.wait(max(0., deadline-time.monotonic()))
            provenance['clock_mapping_end'] = clock.probe(3)
        except InterruptedError as exc:
            self.outcome = 'aborted'
            with self.lock:
                self.state['error'] = str(exc)
        except Exception as exc:
            self.outcome = 'aborted'
            with self.lock:
                self.state['error'] = f'{type(exc).__name__}: {exc}'
            catalog.event('error', 'ROSMASTER запись: '+self.state['error'])
        finally:
            if provider is not None:
                self.operator.set_direct(None)
                self.operator.set_source(None)
                provider.reset()
            http.close()
            if decision is not None:
                try:
                    research.record_event(trial_id, 'finish', time.monotonic_ns(), outcome=self.outcome,
                                          emergency=self.emergency, deviation=self.state.get('error'))
                except Exception as exc:
                    catalog.event('error', 'Журнал пилота: '+str(exc))
            with self.lock:
                self.state['phase'] = 'saving'
            for unused in pool:
                try:unused['writer'].finish('aborted', {'unused_segment': True})
                except Exception as exc:catalog.event('error', 'ROSMASTER пустой сегмент: '+str(exc))
            for index, segment in enumerate(segments):
                try:
                    provenance.update(stop_reason=self.state.get('error') or 'operator marked outcome',
                                      segment_index=index, segments=len(segments), command_source=segment['source'],
                                      emergency_intervention=self.emergency)
                    segment['writer'].finish(self.outcome, provenance)
                    if check:
                        catalog.exclude(segment['writer'].id, True)
                except Exception as exc:
                    with self.lock:
                        self.state['error'] = f'Сохранение: {exc}'
                    catalog.event('error', 'ROSMASTER сохранение: '+str(exc))
            lease.__exit__(None, None, None)
            with self.lock:
                self.state['phase'] = 'idle'
                self.state['outcome'] = self.outcome
                self.state['segments'] = [s['writer'].id for s in segments]

    def sample(self, http, clock):
        response = http.get('/sample.jpg')
        received = time.monotonic_ns()
        response.raise_for_status()
        state = json.loads(response.headers['X-Explorer-Sample'])
        camera = json.loads(response.headers['X-Explorer-Camera'])
        bgr = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError('Неверный JPEG камеры')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape != (480, 640, 3):
            raise RuntimeError(f'Неожиданный размер камеры {rgb.shape}')
        state.update(rgb=rgb, camera=camera, homebrain_received_ns=received,
                     observation_ns=clock.to_homebrain(state['agent_monotonic_ns']),
                     camera_ns=clock.to_homebrain(camera['received_ns']))
        return state

    def frame(self, name, cfg, current, following, len_frames):
        state, action = state_and_action(name, current, following)
        action_ns = following['observation_ns']
        if following['command_source'] == 'policy':
            source = 'policy'
        else:
            source = 'human'
        image = {'image': current['rgb'], 'sequence': current['camera']['sequence'], 'timestamp_ns': current['camera_ns'],
                 'agent_receive_timestamp_ns': current['camera']['received_ns'],
                 'source_timestamp_ns': current['camera']['source_timestamp_ns'],
                 'source_clock': current['camera']['source_clock'], 'homebrain_receive_timestamp_ns': current['homebrain_received_ns'],
                 'timestamp_kind': 'agent_receive_mapped_to_homebrain'}
        base = name == 'rosmaster_dock'
        return {
            'frame_id': len_frames, 'images': {cfg['camera']: image},
            'observation.state': state, 'action': action,
            'observation_timestamp_ns': current['observation_ns'],
            # The command in force at the next sample: decided after this observation.
            'action_send_start_ns': action_ns, 'action_send_end_ns': action_ns,
            'action_time_kind': 'command_in_force_at_next_agent_snapshot', 'action_acknowledged': False,
            'command_source': source, 'teleoperation_phase': 'trial' if source == 'policy' else 'following',
            'measured_response_kind': 'wheel_encoder_velocity' if base else 'servo_position_readback',
            'requested_action': current.get('requested_action') or (current['operator_intent'] if source == 'human' else None),
            'action_after_limits': following['base_after_limits'] if base else following['arm_target'],
            'action_sent': following['base_sent'] if base else following['arm_sent'],
            'action_sent_agent_ns': following['base_sent_ns'] if base else following['arm_sent_ns'],
            'base_requested': following['base_requested'], 'base_guard': following['base_guard'],
            'arm_blocked': following['arm_blocked'],
            'measured_response': following['base_measured'] if base else following['arm_measured'],
            'measured_response_age_s': following['base_measured_age_s'] if base else following['arm_measured_ages_s'],
            'arm_measured_valid': current['arm_measured_valid'],
            'battery_voltage': current['battery_voltage'], 'battery_age_s': current['battery_age_s'],
            'battery_source': 'controller BATTERY frame (native serial)',
            'agent_control_step': current['control_step'], 'next_agent_control_step': following['control_step'],
            'agent_monotonic_ns': current['agent_monotonic_ns'], 'robot_clock': 'rosmaster_monotonic_ns',
            'controller_ready': current['controller_ready'], 'arm_cancel_recoveries': following['arm_cancel_recoveries'],
            'mode': current['mode'], 'decision': current.get('decision'),
        }
