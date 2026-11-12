"""Observer freshness and camera integrity without ROS or devices."""
import json
from types import SimpleNamespace as NS
import numpy as np
from fastapi.testclient import TestClient
from agents.rosmaster.observer import Observer
from agents.rosmaster import agent


def test_missing_and_stale_feedback_never_becomes_assumed_pose(monkeypatch):
    now=1_000_000_000
    monkeypatch.setattr('agents.rosmaster.observer.time.monotonic_ns', lambda:now)
    observer=Observer()
    assert observer.snapshot()['arm_measured'] is None
    observer._arm(NS(position=[90,160,-61,32,90,float('nan')]))
    state=observer.snapshot()
    assert state['arm_measured']==[90,160,-61,32,90,None]
    assert not state['arm_complete'] and state['arm_outside_native_model']==[3]
    assert state['controller_command_readiness']=='not_measured_by_this_observer'
    now+=500_000_001
    assert observer.snapshot()['arm_measured'] is None


def test_padded_bgr_frames_and_stale_jpeg(monkeypatch):
    import pytest
    now=1_000_000_000
    monkeypatch.setattr('agents.rosmaster.observer.time.monotonic_ns',lambda:now)
    observer=Observer()
    pixels=np.array([10,20,30,40,50,60,0,0],dtype=np.uint8)
    observer._image(NS(encoding='bgr8',height=1,width=2,step=8,data=pixels,
                       header=NS(stamp=NS(sec=4,nanosec=7))))
    assert observer.rgb.tolist()==[[[30,20,10],[60,50,40]]]
    data,state=observer.jpeg()
    assert data.startswith(b'\xff\xd8') and state['camera']['source_timestamp_ns']==4_000_000_007
    now+=600_000_000
    with pytest.raises(RuntimeError,match='fresh'):observer.jpeg()


def test_agent_authorization_and_marker_cannot_enable_motors(tmp_path,monkeypatch):
    monkeypatch.setattr(agent,'ROOT',tmp_path)
    (tmp_path/'config').mkdir()
    (tmp_path/'config/agent.json').write_text(json.dumps({'homebrain':'testclient','token':'test-token'}))
    (tmp_path/'stage-d-authorized.json').write_text('{}')
    with TestClient(agent.app) as client:
        assert client.get('/health').status_code==403
        client.headers['X-Explorer-Token']='test-token'
        state=client.get('/health').json()
        assert not state['hardware_open'] and not state['armed']
        assert client.post('/session').status_code==409
        assert client.get('/frame.jpg').status_code==503
        assert client.post('/stop').json()['physical_stop_confirmed'] is False
