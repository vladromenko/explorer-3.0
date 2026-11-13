"""Separate bounded helper segments with shared-world provenance, never auto-import."""
import hashlib
import uuid
from .capture import SimCapture
from . import state_codec


class AgentRecordingView:
    def __init__(self,runner,name):self.runner=runner;self.name=name;self.env=runner.world.views[name].env
    @property
    def manifest(self):return self.runner.world.agent_manifest(self.name)
    def render(self):return self.runner.world.render(self.name)
    def __getattr__(self,key):return getattr(self.env,key)


class FleetCorrections:
    def __init__(self,runner,seed):
        self.runner=runner;self.seed=seed;self.lineage=uuid.uuid4().hex;self.active={};self.segments=[];self.entry_state=None

    def prepare_boundary(self):
        runner=self.runner
        possible=any(m.mode=='policy' and m.counter>=m.cfg.reject_persistence-1 for m in runner.monitors.values())
        possible=possible or any(r['status']=='pending' for r in runner.requests.values())
        possible=possible or (runner.helper_name is not None and runner.helper_name not in self.active)
        if possible:self.entry_state=runner.snapshot()

    def end_completed(self):
        for name in list(self.active):
            if self.runner.world.views[name].done:self.end(name)

    def observe(self,name,action,source):
        if source=='scripted_expert':
            view=AgentRecordingView(self.runner,name)
            if name not in self.active:
                capture=SimCapture(view,self.seed+list(self.runner.world.views).index(name),'scripted_expert')
                if self.entry_state is None:raise RuntimeError('No complete pre-step intervention snapshot')
                snapshot_hash=state_codec.dump(capture.root/'entry-snapshot.json',self.entry_state)
                capture.meta.update(fleet_seed=self.seed,lineage=self.lineage,session=self.lineage,agent=name,segment_role='fleet_correction',
                    training_eligible=False,selection_origin=self.runner.policies[name].metadata['kind'],
                    entry_snapshot_sha256=snapshot_hash,shared_world_sha256=self.runner.world.manifest['xml_sha256'],
                    start_world_time=float(self.runner.world.data.time),expert_privileged=True)
                self.active[name]=capture
            self.active[name].add(view,action)
        elif name in self.active:self.end(name)

    def end(self,name):
        capture=self.active.pop(name);capture.meta['end_world_time']=float(self.runner.world.data.time)
        self.segments.append((name,capture))

    def finish(self,aborted=False):
        for name in list(self.active):self.end(name)
        results=[]
        for name,capture in self.segments:
            view=self.runner.world.views[name]
            outcome='aborted' if aborted else view.result[4]['outcome'] if view.done else 'aborted'
            reason='run_aborted' if aborted else view.result[4]['termination_reason'] if view.done else 'run_horizon_not_task_success'
            root=capture.finish(outcome,reason)
            results.append({'agent':name,'profile':view.env.profile,'spool':str(root),'frames':capture.count,
                'lineage':self.lineage,'outcome':outcome,'termination_reason':reason,'review_required':True,
                'training_eligible':False,'selection_origin':capture.meta['selection_origin']})
        return results
