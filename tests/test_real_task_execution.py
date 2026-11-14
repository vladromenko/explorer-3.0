"""Controller spies only: no real session, policy, simulation, serial or network."""
import json
import threading
import time
import pytest
from explorer3 import catalog,resources
from explorer3.cognitive import real_execution as module,store


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    for m in (catalog,resources,store,module):monkeypatch.setattr(m,'DATA',tmp_path)
    catalog.initialize();store.initialize()
    nodes=[{'id':'a','depends_on':[]},{'id':'b','depends_on':['a']}]
    plan={'graph':{'nodes':nodes},'trials':{k:{'trial_id':k,'profile':'tank_dock'} for k in ('a','b')}}
    monkeypatch.setattr(module,'load',lambda _:plan)
    monkeypatch.setattr(module.trials,'load',lambda i:{'id':i})
    monkeypatch.setattr(module.trials,'require_physical',lambda _:None)
    return plan


class Spy:
    def __init__(self):self.started=[];self.stopped=[];self.value={'active':True};self.heartbeat_count=0
    def start(self,ident,owner,admitted):
        assert admitted();self.started.append(ident);self.value={'active':True}
    def stop(self,reason):self.stopped.append(reason);self.value={'active':False}
    def heartbeat(self):self.heartbeat_count+=1
    def finish(self,outcome):self.finished=outcome
    def status(self):return self.value
    def close_step(self):self.closed=True


def episode(ident='record',outcome='success'):
    with catalog.connect() as db:
        db.execute('INSERT INTO episodes(id,profile,session,scene,outcome,frames,dataset,created,episode_index,category) VALUES(?,?,?,?,?,?,?,?,?,?)',
            (ident,'tank_dock','fixture','fixture',outcome,10,'fixture-no-files',time.time(),0,'rollout'))


def test_confirmation_and_existing_physical_gate_precede_dispatch(runtime,monkeypatch):
    spy=Spy();ex=module.Executor(spy)
    with pytest.raises(PermissionError):ex.start('plan')
    def denied(_):raise PermissionError('Tank incident active')
    monkeypatch.setattr(module.trials,'require_physical',denied)
    with pytest.raises(PermissionError,match='incident'):ex.start('plan',operator_confirmed=True)
    assert not spy.started and ex.claim is None


def test_unknown_exit_never_starts_dependent_robot(runtime):
    spy=Spy();ex=module.Executor(spy);ident=ex.start('plan',operator_confirmed=True)
    spy.value={'active':False,'outcome':'success','episode_id':'missing'}
    ex.tick()
    assert spy.started==['a'] and ex.claim is None
    assert store.mission(ident)['task_outcome']=='unknown'


def test_verified_step_requires_saved_episode_and_explicit_advance(runtime):
    spy=Spy();ex=module.Executor(spy);ex.start('plan',operator_confirmed=True)
    with pytest.raises(ValueError):ex.advance()
    episode();ex.finish_step('success')
    spy.value={'active':False,'outcome':'success','episode_id':'record','error':None};ex.tick()
    assert spy.started==['a'] and ex.state['stage']=='ready_for_next'
    ex.tick();ex.tick()
    assert ex.claim and spy.closed and len(ex.state['node_outcomes'])==1
    ex.heartbeat();ex.advance();assert spy.started==['a','b']
    ex.cancel();assert ex.claim is None and spy.stopped


def test_hidden_browser_and_restart_do_not_resume(runtime):
    spy=Spy();ex=module.Executor(spy);ex.start('plan',operator_confirmed=True)
    ex.last_seen-=3;ex.tick();assert ex.claim is None and spy.started==['a']
    ex.start('plan',operator_confirmed=True);store.recover_interrupted();ex.tick()
    assert ex.claim is None and spy.started==['a','a']


def test_stop_during_long_preflight_cannot_dispatch(runtime,monkeypatch):
    entered=threading.Event();release=threading.Event();spy=Spy();ex=module.Executor(spy);errors=[]
    def gate(_):entered.set();assert release.wait(3)
    monkeypatch.setattr(module.trials,'require_physical',gate)
    def start():
        try:ex.start('plan',operator_confirmed=True)
        except Exception as exc:errors.append(str(exc))
    thread=threading.Thread(target=start);thread.start();assert entered.wait(1)
    ex.cancel('STOP during hash check');release.set();thread.join(3)
    assert not thread.is_alive() and not spy.started and errors


def test_stop_can_interrupt_adapter_start_without_waiting_for_network(runtime):
    entered=threading.Event();release=threading.Event();spy=Spy();ex=module.Executor(spy);errors=[]
    def delayed(ident,owner,admitted):
        entered.set();assert release.wait(3)
        if not admitted():raise ValueError('Cancelled before device enable')
        spy.started.append(ident)
    spy.start=delayed
    def start():
        try:ex.start('plan',operator_confirmed=True)
        except Exception as exc:errors.append(str(exc))
    thread=threading.Thread(target=start);thread.start();assert entered.wait(1)
    ex.cancel('STOP');release.set();thread.join(3)
    assert not thread.is_alive() and not spy.started and spy.stopped and errors
