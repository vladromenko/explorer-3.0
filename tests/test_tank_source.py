"""Tank observation/command contract without networking or motors."""
import json
import threading
import time
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from explorer3 import tank, local_work, trials


@pytest.fixture
def source(monkeypatch):
    requests=[]
    accepted=[0.,0.]
    jpeg=cv2.imencode('.jpg',np.zeros((400,640,3),np.uint8))[1].tobytes()
    def route(request):
        now=time.monotonic_ns()
        if request.url.path=='/health':return httpx.Response(200,json={'monotonic_ns':now})
        if request.url.path=='/session':return httpx.Response(200,json={'lease':'test-only'})
        if request.url.path=='/stop':return httpx.Response(200,json={})
        if request.url.path=='/frame.jpg':
            info={'sequence':1,'source_timestamp_ns':123,'source_clock':'robot_ros', 'received_ns':now-1000}
            status={'monotonic_ns':now,'action':list(accepted),'battery_voltage':8.,'battery_received_ns':now,
                    'imu':None,'imu_received_ns':None}
            return httpx.Response(200,content=jpeg,headers={'X-Explorer-Camera':json.dumps(info),'X-Explorer-Observation':json.dumps(status)})
        value=json.loads(request.content);requests.append(value)
        accepted[:]=[max(-.2,min(.2,value['v'])),max(-.5,min(.5,value['w']))]
        return httpx.Response(200,json={'action':list(accepted),'send_start_ns':now,'send_end_ns':now+1,'motor_targets_rps':[]})
    monkeypatch.setattr(tank,'client',lambda:httpx.Client(base_url='http://test',transport=httpx.MockTransport(route)))
    src=tank.TankSource(SimpleNamespace(base_action=lambda:[.2,.5]))
    src.start(explicit_user_start=True)
    yield src,requests
    src.close()


def test_record_and_direct_action_have_causal_real_channel_contract(source):
    src,requests=source
    row=src.sample(0)
    assert row['observation.state']==[0.,0.] and row['action']==[.2,.5]
    assert row['observation_timestamp_ns']<=row['action_send_start_ns']<=row['action_send_end_ns']
    assert row['images']['front']['image'].shape==(400,640,3)
    assert not row['action_acknowledged'] and row['agent_accepted_command']
    observed=src.observe();assert observed['observation.state']==[.2,.5]
    action=src.send([.4,-.8],observed)
    assert action['action']==[.2,-.5] and action['requested_action']==[.4,-.8]
    assert [r['sequence'] for r in requests]==[0,1]
    assert observed['images']['front']['reused']


def test_policy_wait_sends_only_zero_and_stop_prevents_result_execution(source):
    src,requests=source
    stopped=threading.Event();src.stop_requested=stopped.is_set
    class Provider:
        def needs_inference(self):return True
        def next(self,observation):
            time.sleep(.22);stopped.set();return [.2,.5]
    with pytest.raises(InterruptedError):src.plan_action(Provider(),src.observe())
    assert len(requests)>=2
    assert all(r['v']==0 and r['w']==0 for r in requests)
    with pytest.raises(InterruptedError):src.send([.2,.5])


def test_cached_action_does_not_insert_neutral_between_every_frame(source):
    src,requests=source
    provider=SimpleNamespace(needs_inference=lambda:False,next=lambda observation:[.1,.2])
    assert src.plan_action(provider,src.observe())==[.1,.2]
    assert not requests


def test_tank_permission_does_not_enable_other_robots(tmp_path,monkeypatch):
    monkeypatch.setattr(local_work,'DATA',tmp_path)
    from explorer3 import fleet_config, permission_migration
    permission_migration.migrate(tmp_path)
    fleet_config.update_robot('so101',enabled=False,root=tmp_path)
    fleet_config.update_robot('rosmaster',enabled=False,root=tmp_path)
    assert local_work.hardware_allowed('tank')
    assert not local_work.hardware_allowed('so101') and not local_work.hardware_allowed('rosmaster')


def test_tank_trial_still_requires_plan_specific_permission(tmp_path,monkeypatch):
    monkeypatch.setattr(trials,'DATA',tmp_path)
    monkeypatch.setattr(local_work,'require_hardware_mode',lambda robot:None)
    (tmp_path/'validation').mkdir();(tmp_path/'configs').mkdir()
    plan={'id':'1'*32,'profile':'tank_dock'}
    with pytest.raises(PermissionError,match='ручное управление'):trials.require_physical(plan)
    (tmp_path/'validation/tank-manual-control-pass.json').write_text('{"operator_confirmed_stop":true}')
    with pytest.raises(PermissionError,match='отдельный физический блок'):trials.require_physical(plan)


def test_handoff_after_reboot_preserves_history_and_captures_current_services(tmp_path,monkeypatch):
    from agents.tank import handoff
    state=tmp_path/'handoff.json'
    previous={'boot_id':'previous','restored':False,'services':{'start_node.service':True,'wifi.service':True}}
    state.write_text(json.dumps(previous))
    monkeypatch.setattr(handoff,'ROOT',tmp_path);monkeypatch.setattr(handoff,'STATE',state)
    monkeypatch.setattr(handoff,'BOOT','current')
    monkeypatch.setattr(handoff,'remote_json',lambda name:{'stage':'C'})
    monkeypatch.setattr(handoff,'api',lambda path:{'hardware_open':False})
    monkeypatch.setattr(handoff,'active',lambda service:False)
    monkeypatch.setattr(handoff,'factory_launches',lambda:[])
    monkeypatch.setattr(handoff,'owners',lambda:[])
    markers=[];monkeypatch.setattr(handoff,'marker',markers.append)
    result=handoff.enter()
    assert result['boot_id']=='current' and not any(result['services'].values())
    assert json.loads((tmp_path/'handoff-previous-previous.json').read_text())==previous
    assert markers==[True]


def test_failed_stop_blocks_future_homebrain_sessions(tmp_path,monkeypatch):
    monkeypatch.setattr(tank,'DATA',tmp_path)
    monkeypatch.setattr(tank,'client',lambda:httpx.Client(base_url='http://test',transport=httpx.MockTransport(
        lambda request:httpx.Response(409,json={'detail':'STOP not delivered: serial EIO'}))))
    src=tank.TankSource(SimpleNamespace())
    errors=src.close()
    assert 'Выключите питание' in errors[0]
    saved=json.loads((tmp_path/'validation/tank-motion-incident.json').read_text())
    assert saved['blocked'] and saved['physical_stop_confirmed'] is False
