"""Thin binding to the already used Explorer trial sessions, not a controller."""
from explorer3 import trials
from explorer3.profiles import profile


class ExistingTrials:
    def __init__(self,session):
        self.session=session;self.robot=None;self.owner=None;self.plan=None

    def start(self,trial_id,owner,admitted):
        plan=trials.load(trial_id);robot=profile(plan['profile'])['robot']
        if not admitted():raise PermissionError('Mission cancelled before opening a robot')
        self.robot=robot;self.owner=owner;self.plan=plan
        if robot=='rosmaster':
            from explorer3 import rosmaster_console as console
            from explorer3.rosmaster_control import ZERO
            from explorer3.rosmaster_session import control_profile
            operator=console.operator
            if operator.lease or console.recorder.active():raise PermissionError('ROSMASTER is owned by another session')
            generation=operator.generation
            console.ensure_agent()
            if not admitted():raise PermissionError('Mission cancelled during agent startup')
            health=console.request('/health')
            if not health.get('hardware_open'):console.request('/acquire',{},45)
            if not admitted():raise PermissionError('Mission cancelled during ownership transfer')
            operator.input(owner,ZERO.copy())
            operator.enable(owner,control_profile(plan['profile']),'joints',True,'policy',expected_generation=generation)
            if not admitted():
                operator.stop('Mission cancelled before trial');raise PermissionError('Cancelled')
            console.recorder.start(owner,plan['profile'],'team-'+owner[:8],collection='team-'+owner,
                plan=plan)
        else:
            control=self.session.control
            if control.owner or self.session.thread and self.session.thread.is_alive():raise PermissionError('Robot session is occupied')
            control.select(robot);generation=control.generation
            if not admitted():raise PermissionError('Mission cancelled before trial')
            self.session.start_trial(owner,trial_id,'team-'+owner[:8],expected_generation=generation)

    def heartbeat(self):
        if self.robot=='rosmaster':
            from explorer3.rosmaster_control import operator,ZERO
            if self.owner:operator.input(self.owner,ZERO.copy())
        elif self.robot:self.session.control.heartbeat(self.owner)

    def finish(self,outcome):
        if self.robot=='rosmaster':
            from explorer3.rosmaster_console import recorder,operator
            recorder.finish(outcome)
            operator.set_direct(None)
        else:self.session.finish(outcome)

    def close_step(self):
        if self.robot=='rosmaster':
            from explorer3.rosmaster_console import operator
            if operator.owner==self.owner:operator.stop('Шаг сохранён; ожидание следующего подтверждения')

    def status(self):
        if self.robot=='rosmaster':
            from explorer3.rosmaster_console import recorder
            value=recorder.status();segments=value.get('segments',[])
            return {'active':recorder.active(),'outcome':value.get('outcome'),'error':value.get('error'),
                'episode_id':segments[-1] if len(segments)==1 else None}
        return {'active':bool(self.session.thread and self.session.thread.is_alive()),
            'outcome':self.session.outcome,'error':self.session.error,'episode_id':self.session.last_episode_id}

    def stop(self,reason):
        if self.robot=='rosmaster':
            from explorer3.rosmaster_console import recorder,operator
            if operator.owner in (None,self.owner):
                recorder.mark_emergency();recorder.finish('aborted',emergency=True);operator.stop(reason)
        elif self.robot:
            if self.session.control.owner in (None,self.owner):self.session.control.stop(reason)
