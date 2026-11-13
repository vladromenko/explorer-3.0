"""One randomized correction per Tank trial; this module never sends commands."""
import threading
import time
from . import research
from .profiles import profile


def accepted_study(ident, plan):
    study = next((s for s in research.studies() if s['id'] == ident), None)
    if plan['profile'] != 'tank_dock' or plan['kind'] != 'policy':
        raise ValueError('Пилот Tank требует испытания политики tank_dock')
    if not study or study['profile'] != plan['profile'] or study['policy_sha256'] != plan['checkpoint_sha256']:
        raise ValueError('Пилот должен соответствовать задаче и выбранному checkpoint')
    if not study.get('timing_accepted'):
        raise ValueError('Сначала проверьте точку решения и задержки на странице исследования')
    if study['decision_after_s'] + max(study['assigned_delays_s']) + study['correction_limit_s'] >= plan['max_duration_s']:
        raise ValueError('Лимит испытания должен оставлять время политике после коррекции')
    return study


class TankPilot:
    # Separate writers prevent ACT action chunks crossing source boundaries.
    sources = {'policy': 'policy', 'hold': 'safety', 'correction': 'human', 'resumed': 'policy'}

    def __init__(self, study, trial_id, session_id, scene):
        self.study, self.trial_id = study, trial_id
        self.session_id, self.scene = session_id, scene
        self.history = []
        self.decision = None
        self.phase = 'policy'
        self.human_started = False
        self.correction_done = threading.Event()
        self.emergency = False
        self.deadline_ns = None

    def tick(self, now_ns, started_ns):
        previous = self.phase
        if self.decision is None and self.history and now_ns-started_ns >= self.study['decision_after_s']*1e9:
            online = research.features(self.history, now_ns, profile('tank_dock')['feature_scales'])
            self.decision = research.assign(self.study['id'], self.trial_id, self.session_id, self.scene,
                                           now_ns, online, timing_accepted=True)
            if self.decision['assigned']:
                research.record_event(self.trial_id, 'request', now_ns)
                self.phase = 'hold'
                self.deadline_ns = now_ns + int(self.decision['assigned_delay_s']*1e9)
        if self.phase == 'hold' and now_ns >= self.deadline_ns:
            research.record_event(self.trial_id, 'operator_notified', now_ns, source='server_dispatch')
            self.phase = 'correction'
            # Fixed assignment deadline: notification/network delay never extends the human budget.
            self.deadline_ns = self.decision['deadline_ns']
        if self.phase == 'correction' and (now_ns >= self.deadline_ns or self.correction_done.is_set()):
            self.end_correction(now_ns)
            self.phase = 'resumed'
        return self.phase != previous

    def end_correction(self, now_ns):
        if not self.human_started:
            research.record_event(self.trial_id, 'human_start', now_ns, performed=False)
        research.record_event(self.trial_id, 'human_end', now_ns)

    def observe(self, sample):
        if self.phase == 'correction' and not self.human_started and any(abs(v) > 1e-6 for v in sample['action']):
            now = sample['action_send_end_ns']
            research.record_event(self.trial_id, 'operator_response', now)
            research.record_event(self.trial_id, 'human_start', now, performed=True)
            self.human_started = True
        image = sample['images']['front']
        self.history.append({k: v for k, v in sample.items() if k != 'images'} | {
            'camera': {'timestamp_ns': image['timestamp_ns']}, 'state_valid': False})
        self.history[:] = self.history[-120:]

    def finish(self, outcome, reason):
        if self.decision:
            now = time.monotonic_ns()
            if self.phase == 'correction':
                self.end_correction(now)
            research.record_event(self.trial_id, 'finish', now, outcome=outcome,
                                  emergency=self.emergency, deviation=reason if outcome == 'aborted' else None)

    def status(self):
        return {'phase': self.phase, 'assigned': self.decision['assigned'] if self.decision else None,
                'remaining_s': max(0., (self.deadline_ns-time.monotonic_ns())/1e9) if self.deadline_ns else None}
