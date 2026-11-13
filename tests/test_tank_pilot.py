"""Trial transitions and audit boundaries, without network or motors."""
import threading
from types import SimpleNamespace
import pytest
from explorer3 import catalog, research, tank_pilot, session as sessions, resources, trials
from explorer3.control import Control


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, 'DATA', tmp_path)
    monkeypatch.setattr(catalog, 'prepare', lambda: None)
    catalog.initialize();research.initialize()
    ident=research.create_study('tank_dock','a'*64,delays=(1.,),correction_limit=2.)
    research.accept_timing(ident,1.,[1.])
    monkeypatch.setattr(research.secrets,'randbelow',lambda n:1)
    return next(s for s in research.studies() if s['id']==ident)


def sample(now,action):
    return {'observation_timestamp_ns':now-2,'action_send_start_ns':now-1,'action_send_end_ns':now,
            'action':action,'observation.state':[0.,0.],
            'measured_response_kind':'previous_command_estimate_no_wheel_feedback',
            'images':{'front':{'timestamp_ns':now-3}}}


def test_one_assignment_delay_budget_and_missing_physical_features(study,monkeypatch):
    p=tank_pilot.TankPilot(study,'trial','session','scene')
    p.observe(sample(500_000_000,[.2,0.]))
    assert p.tick(1_000_000_000,0) and p.phase=='hold'
    assert p.decision['features']['response_change'] is None
    assert p.decision['features']['tracking_error'] is None
    assert not p.tick(1_500_000_000,0)
    assert p.tick(2_000_000_000,0) and p.phase=='correction'
    p.observe(sample(2_100_000_000,[0.,.5]))
    assert p.tick(4_000_000_000,0) and p.phase=='resumed'
    assert not p.tick(8_000_000_000,0)
    monkeypatch.setattr(tank_pilot.time,'monotonic_ns',lambda:9_000_000_000)
    p.finish('success',None)
    rows=research.records();assert len(rows)==1
    events={e['kind']:e for e in rows[0]['events']}
    assert events['human_end']['time_ns']-events['operator_notified']['time_ns']==2_000_000_000
    assert events['human_start']['performed'] is True
    assert rows[0]['outcome']=='success'


def test_no_help_and_early_finish_do_not_fabricate_human_motion(study,monkeypatch):
    monkeypatch.setattr(research.secrets,'randbelow',lambda n:0)
    p=tank_pilot.TankPilot(study,'nohelp','session','scene')
    p.observe(sample(500_000_000,[.2,0.]));p.tick(1_000_000_000,0)
    assert p.phase=='policy' and p.decision['assigned']==0
    monkeypatch.setattr(research.secrets,'randbelow',lambda n:1)
    q=tank_pilot.TankPilot(study,'help','session','scene')
    q.observe(sample(500_000_000,[.2,0.]));q.tick(1_000_000_000,0);q.tick(2_000_000_000,0)
    q.correction_done.set();q.tick(2_100_000_000,0)
    assert q.phase=='resumed' and not q.human_started
    row=next(r for r in research.records() if r['episode_id']=='help')
    assert next(e for e in row['events'] if e['kind']=='human_start')['performed'] is False
    plan={'profile':'tank_dock','kind':'policy','checkpoint_sha256':'b'*64,'max_duration_s':60}
    with pytest.raises(ValueError,match='checkpoint'):tank_pilot.accepted_study(study['id'],plan)


def test_session_segments_are_separate_and_stop_result_is_immutable(study,tmp_path,monkeypatch):
    monkeypatch.setattr(resources,'DATA',tmp_path)
    from explorer3 import tank_handoff
    monkeypatch.setattr(tank_handoff,'invoke',lambda _:None)
    monkeypatch.setattr(trials,'require_physical',lambda _:None)
    writers=[];commands=[];phases=iter(['policy','hold','correction','resumed'])
    class Writer:
        def __init__(self,*args,category):self.category=category;self.rows=[];writers.append(self)
        def add(self,row):self.rows.append(row)
        def finish(self,outcome,provenance):self.outcome=outcome
    monkeypatch.setattr(sessions,'EpisodeWriter',Writer)
    monkeypatch.setattr(trials,'PolicyActions',lambda _:SimpleNamespace(
        next=lambda _: [.2,0.],reset=lambda:None,decision_info={}))
    class Pilot:
        sources=tank_pilot.TankPilot.sources
        decision=None;human_started=True
        def __init__(self,*args):self.phase='policy'
        def tick(self,*args):self.phase=next(phases);return True
        def observe(self,row):pass
        def finish(self,*args):pass
    monkeypatch.setattr(tank_pilot,'TankPilot',Pilot)
    c=Control();c.robot='tank';c.owner='test';c.expires=float('inf');c.stop_event.clear()
    s=sessions.Session(c);s.study=study;s.trial_plan={'id':'prepared-plan','kind':'policy','max_duration_s':60}
    def send(action,observation):
        commands.append(action)
        if len(commands)==4:
            s.finish('success');s.finish('aborted','Disconnect during saving')
        return {'action':action,'action_send_start_ns':0,'action_send_end_ns':1}
    source=SimpleNamespace(start=lambda **kwargs:{},begin_episode=lambda:None,
        observe=lambda:{'observation.state':[0.,0.],'images':{}},
        plan_action=lambda p,o:p.next(o),send=send,close=lambda:[])
    monkeypatch.setattr(sessions,'TankSource',lambda *args,**kwargs:source)
    monkeypatch.setattr(c,'base_action',lambda:[0.,.5])
    s._run('tank_dock','test','scene')
    assert s.error is None and s.outcome=='success'
    assert commands==[[.2,0.],[0.,0.],[0.,.5],[.2,0.]]
    assert [w.rows[0]['command_source'] for w in writers]==['policy','safety','human','policy']
    assert [w.category for w in writers]==['rollout','rollout','correction','rollout']
