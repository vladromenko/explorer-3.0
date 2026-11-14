import json
from types import SimpleNamespace
import pytest
from explorer3 import handoff, fleet_config, resources


@pytest.fixture
def rig(tmp_path,monkeypatch):
    monkeypatch.setattr(handoff,'DATA',tmp_path)
    monkeypatch.setattr(resources,'DATA',tmp_path)
    value={'version':1,'robots':{'so101':{'enabled':True,'owner':'legacy_8020'},
        'tank':{'enabled':True,'mode':'floor'},'rosmaster':{'enabled':True}},'training_authorized':False}
    fleet_config.atomic_write(fleet_config.path(tmp_path),value)
    state={'active':True,'calls':[],'ready':True}
    def systemctl(action,*args,**kw):
        state['calls'].append(action)
        if action=='stop':state['active']=False
        if action=='start':state['active']=True
        return SimpleNamespace(returncode=0 if state['active'] else 3)
    monkeypatch.setattr(handoff,'systemctl',systemctl)
    monkeypatch.setattr(handoff,'legacy_ready',lambda:state['ready'])
    device=tmp_path/'fake-device';device.touch()
    return SimpleNamespace(owner=None),[str(device)],state,tmp_path


def test_explicit_handover_repeat_and_verified_return(rig):
    control,paths,state,root=rig
    assert handoff.enter(control,paths,confirmed=True)['owner']=='explorer'
    assert not state['active']
    handoff.enter(control,paths,confirmed=True)
    assert json.loads((root/'handoff.json').read_text())['was_active']
    closed=[]
    assert handoff.restore(control,close_devices=lambda:closed.append(True))['http_8020']
    assert closed==[True] and state['active']
    assert fleet_config.load(root)['robots']['so101']['owner']=='legacy_8020'
    assert set(p.name for p in root.iterdir())=={'configs','locks','handoff.json','fake-device'}


def test_missing_confirmation_does_not_touch_service(rig):
    c,p,s,r=rig
    with pytest.raises(PermissionError):handoff.enter(c,p)
    assert s['calls']==[]


def test_busy_device_restores_old_service_without_permission(rig,monkeypatch):
    c,p,s,r=rig
    monkeypatch.setattr(resources,'owners',lambda _: [{'pid':123}])
    with pytest.raises(resources.ResourceBusy):handoff.enter(c,p,confirmed=True)
    assert s['active']
    assert fleet_config.load(r)['robots']['so101']['owner']=='legacy_8020'


def test_return_requires_closed_own_handles_and_live_http(rig):
    c,p,s,r=rig
    handoff.enter(c,p,confirmed=True)
    with resources.ResourceLease('so101-leader'):
        with pytest.raises(resources.ResourceBusy):handoff.restore(c)
    assert not s['active']
    assert not fleet_config.robot_allowed('so101',r)[0]
    s['ready']=False
    with pytest.raises(RuntimeError,match='не отвечает'):handoff.restore(c)
    assert not json.loads((r/'handoff.json').read_text())['restored']
    s['ready']=True
    handoff.restore(c)


def test_active_session_and_disabled_robot_refused(rig):
    c,p,s,r=rig
    c.owner='someone'
    with pytest.raises(RuntimeError):handoff.enter(c,p,confirmed=True)
    c.owner=None
    fleet_config.update_robot('so101',enabled=False,root=r)
    with pytest.raises(PermissionError):handoff.enter(c,p,confirmed=True)
    assert s['calls']==[]
