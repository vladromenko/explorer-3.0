"""Hardware-free checks for the deployed manual control methods and browser contract."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from cubebot import config
from cubebot.arm_model import ArmModel
from cubebot.arm_control import CartesianJog
from cubebot.base_control import VelocityShaper, ObstacleGuard

ROOT = Path(config.__file__).parent

def method(file, cls, name):
    tree = ast.parse((ROOT / file).read_text())
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT / file), 'exec'), scope)
    return scope[name]

class IO:
    def __init__(self, pose):
        self._arm_cmd = list(pose); self.measured = list(pose); self.sent = []; self.drives = []
    def measured_arm(self, **kw): return self.measured
    def control_seed(self, **kw): return self.measured
    def arm_command(self): return self._arm_cmd
    def drive(self, *args, **kw): self.drives.append((args,kw))
    def move_arm(self, cmd, **kw):
        assert kw['check'] is True
        self.sent.append(cmd); self._arm_cmd = list(cmd); self.measured = list(cmd)

def setup():
    cfg=config.load(); arm=ArmModel(cfg['arm']['tcp_offset_m']); pose=cfg['arm']['search_pose']
    c=SimpleNamespace(cfg=cfg,io=IO(pose),jog=CartesianJog(arm,cfg,pose),arm_mode='cartesian',
      _arm_input_active=False,_arm_tracking_blocked=False,last_input=1.,
      input=dict(drive=[0,0,0],arm=[0,0,0],pitch=0,roll=0,grip=0,slow=False))
    return c,method('web/server.py','Console','_manual_step')

def test_every_computed_command_is_sent_no_timer_drops():
    c,tick=setup(); c.input['grip']=1
    for _ in range(16): tick(c,.05,1.)
    assert len(c.io.sent)==16
    assert c.io.sent[-1]==np.round(c.jog.servo).tolist()

def test_tracking_failure_requires_release_even_after_feedback_recovers():
    c,tick=setup(); c.input['grip']=1; tick(c,.05,1.)
    c.io.measured[0]+=20; tick(c,.05,1.); n=len(c.io.sent)
    c.io.measured=list(c.io._arm_cmd); tick(c,.05,1.)
    assert len(c.io.sent)==n and c._arm_tracking_blocked
    c.input['grip']=0; tick(c,.05,1.)
    c.input['grip']=1; tick(c,.05,1.)
    assert len(c.io.sent)==n+1

def test_new_gesture_uses_real_feedback_and_stale_input_stops_arm():
    c,tick=setup(); c.io.measured[0]-=5; c.input['grip']=1; tick(c,.05,1.)
    assert c.io.sent[-1][0]==85
    n=len(c.io.sent); tick(c,.05,2.)
    assert len(c.io.sent)==n and c.io.drives[-1][0]==(0,0,0)

def test_gripper_roll_bypass_ik():
    c,_=setup(); c.jog.arm.ik=lambda *a,**kw: pytest.fail('IK should not run')
    old=c.jog.servo[:4].copy()
    cmd=c.jog.step(.05,[0,0,0],roll_rate=1,grip_rate=1)
    assert cmd and np.array_equal(c.jog.servo[:4],old)

@pytest.mark.parametrize('axis',[0,1,2,3,4,5])
def test_direct_joints_independent(axis):
    c,_=setup(); c.jog.reset([90,90,90,90,90,60]); old=c.jog.servo.copy()
    rates=[0]*6;rates[axis]=1
    cmd=c.jog.step_joints(.05,rates)
    assert cmd and np.count_nonzero(c.jog.servo-old)==1
    assert c.jog.arm.collision_free(cmd)

def test_cartesian_local_solver_progress_and_step_bound():
    c,_=setup(); c.jog.reset([40,57,93,119,90,30]); start=c.jog.target.copy()
    for _ in range(20):
        old=c.jog.servo.copy(); c.jog.step(.05,[0,0,-1])
        assert max(abs(c.jog.servo[:4]-old[:4]))<=3.00001
    assert c.jog.target[2]<start[2]-.01

def test_timing_gap_does_not_make_a_large_step():
    c,tick=setup();c.input['grip']=1;tick(c,.5,1.)
    assert not c.io.sent and c._arm_tracking_blocked

def test_manual_speed_profile_not_inherited_by_mission():
    c,_=setup();io=SimpleNamespace(cfg=c.cfg,shaper=VelocityShaper(c.cfg['base']),now=lambda:0)
    drive=method('robot_io.py','BaseIO','drive')
    drive(io,.3,0,0,manual=True)
    for _ in range(30): out=io.shaper.step(io._desired,.05)
    assert out[0]==.3
    drive(io,.3,0,0)
    for _ in range(30): out=io.shaper.step(io._desired,.05)
    assert out[0]==.1 and c.cfg['base']['max_vx']==.1
    wall=np.stack([np.full(50,.2),np.linspace(-.3,.3,50)],axis=1)
    assert ObstacleGuard(c.cfg['base']).limit((.3,0,0),wall)[0]==0

def test_relative_mirror_first_frame_does_not_move_gripper_or_joints():
    from cubebot.mirror import MirrorMapper, load, JOINTS
    c,_=setup(); m=load();m.update(relative=True,smoothing=1.)
    mp=MirrorMapper(c.jog.arm,c.cfg['arm'],m,[90,90,90,90,90,60])
    sample=dict(zip(JOINTS,[20,-40,60,30,-80,27]))
    mp.update(sample,1.)
    assert mp.step(.05,1.) is None
    mp.update(dict(sample,gripper=37),1.05)
    assert mp.target[5]==pytest.approx(69.5)
    assert np.array_equal(mp.target[:5],[90]*5)

def test_mirror_stream_timeout_disarms_instead_of_waiting_to_resume():
    from cubebot.mirror import MirrorMapper,load,JOINTS
    c,_=setup();c.mode='mirror';c.log=lambda s:None
    m=load();m['relative']=True
    c.mirror=MirrorMapper(c.jog.arm,c.cfg['arm'],m,c.io.arm_command())
    c.mirror.update(dict(zip(JOINTS,[0]*6)),1.)
    c.stop_all=lambda:setattr(c,'mode','manual')
    method('web/server.py','Console','_mirror_step')(c,.05,2.)
    assert c.mode=='manual' and not c.io.sent
    c.mirror.update(dict(zip(JOINTS,[2]*6)),2.1)
    assert c.mode=='manual'

def test_mirror_tracking_fault_disarms():
    from cubebot.mirror import MirrorMapper,load
    c,_=setup();c.mode='mirror';c.log=lambda s:None
    c.mirror=MirrorMapper(c.jog.arm,c.cfg['arm'],load(),c.io.arm_command());c.mirror.last_msg=1.
    c.io.measured[0]+=30;c.stop_all=lambda:setattr(c,'mode','manual')
    method('web/server.py','Console','_mirror_step')(c,.05,1.)
    assert c.mode=='manual' and not c.io.sent
