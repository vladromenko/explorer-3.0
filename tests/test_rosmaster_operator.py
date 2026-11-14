"""Homebrain operator loop with a fake agent: no network, no robot."""
import time
import httpx
import pytest
from explorer3 import rosmaster_control as rc


class Agent:
    def __init__(self):
        self.packets=[];self.fail=0;self.reject=None;self.stops=0;self.reason='Battery telemetry stale (>30 s)'
    def post(self,path,json=None):
        if self.fail:
            self.fail-=1;raise httpx.ReadTimeout('late')
        self.packets.append(json)
        if self.reject:return httpx.Response(409,json={'detail':self.reject})
        return httpx.Response(200,json={'armed':True,'reason':None})
    def close(self):pass


@pytest.fixture
def op(monkeypatch,tmp_path):
    agent=Agent()
    monkeypatch.setattr(rc,'DATA',tmp_path)
    monkeypatch.setattr('explorer3.local_work.hardware_allowed',lambda *_:True)
    monkeypatch.setattr(rc,'connection',lambda timeout=2:agent)
    def request(path,value=None,timeout=2):
        if path=='/stop':agent.stops+=1
        return {}
    monkeypatch.setattr(rc,'request',request)
    monkeypatch.setattr(rc.Operator,'agent_reason',lambda self:agent.reason)
    o=rc.Operator();o.agent=agent
    yield o
    o.done.set();o.thread.join(1)


def start(o,intent=None):
    with o.lock:
        o.owner='a';o.lease='L';o.inputs['a']=(time.monotonic(),intent or {**rc.ZERO,'drive':[.5,0,0]})


def feed(o,seconds,intent=None):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        o.input('a',intent or {**rc.ZERO,'drive':[.5,0,0]});time.sleep(.02)


def test_single_late_reply_does_not_end_session(op):
    start(op);feed(op,.3);op.agent.fail=1;feed(op,.4)
    assert op.lease=='L' and op.agent.stops==0
    assert op.stats['timeouts']==1


def test_repeated_timeouts_stop(op):
    start(op);feed(op,.2);op.agent.fail=1000;feed(op,1.3)
    assert op.lease is None and 'Нет ответа агента' in op.error and op.agent.stops>=1


def test_browser_pause_holds_zero_then_stops(op):
    start(op);feed(op,.2);time.sleep(.5)
    assert op.lease=='L'
    assert op.agent.packets[-1]['intent']['drive']==[0,0,0]
    time.sleep(2.8)
    assert op.lease is None and 'heartbeat' in op.error


def test_agent_reason_is_shown_not_masked(op):
    start(op);feed(op,.2);op.agent.reject='Сессия остановлена';feed(op,.2)
    assert op.lease is None
    assert 'Battery telemetry stale' in op.error
    log=(rc.DATA/'logs/rosmaster-control.jsonl').read_text()
    assert 'agent' in log and 'Battery' in log


def test_closed_scope_stops_using_cached_command_connection(op,monkeypatch):
    start(op);feed(op,.15);before=len(op.agent.packets)
    monkeypatch.setattr('explorer3.local_work.hardware_allowed',lambda *_:False)
    time.sleep(.15)
    assert op.lease is None and len(op.agent.packets)<=before+1
    count=len(op.agent.packets);time.sleep(.1);assert len(op.agent.packets)==count


def test_compute_gate_precedes_robot_session_and_stop_releases_it(op,monkeypatch,tmp_path):
    from explorer3 import resources
    monkeypatch.setattr(resources,'DATA',tmp_path)
    monkeypatch.setattr(rc,'enabled',lambda:True)
    calls=[]
    def request(path,value=None,timeout=2):
        calls.append(path);return {'lease':'fixture'}
    monkeypatch.setattr(rc,'request',request)
    op.input('browser',rc.ZERO.copy())
    with resources.ResourceLease('physical-activity'):
        with pytest.raises(resources.ResourceBusy):op.enable('browser','rosmaster_dock','xyz',False)
    assert not calls
    op.enable('browser','rosmaster_dock','xyz',False)
    with pytest.raises(resources.ResourceBusy):
        with resources.ResourceLease('physical-activity'):pass
    op.stop('fixture STOP')
    with resources.ResourceLease('physical-activity'):pass
    op.input('browser',rc.ZERO.copy());op.enable('browser','rosmaster_dock','xyz',False)
    assert op.lease=='fixture';op.stop('done')
    generation=op.generation;op.stop('late STOP');calls.clear()
    op.input('browser',rc.ZERO.copy())
    with pytest.raises(PermissionError):op.enable('browser','rosmaster_dock','xyz',False,expected_generation=generation)
    assert not calls
