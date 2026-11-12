import ast
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cubebot import config, mirror
from cubebot.arm_model import ArmModel


def method(file, clsname, name):
    source=Path(config.__file__).parent/file
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==clsname)
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'math':math,'time':time,'json':json,'np':np,'ArmJoints':SimpleNamespace,
           '__name__':'cubebot.ros_io','__package__':'cubebot'}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
    return scope[name]

def io_fixture():
    io=SimpleNamespace(_servo_fb=([90.,90.,90.,10.,90.,math.nan],time.monotonic()),
       _gripper_command=30., _arm_cmd=[90.,90.,90.,10.,90.,30.],cfg=config.load(),
       measured_arm=lambda **kw:None)
    io.feedback_state=lambda max_age=.25:method('ros_io.py','RosRobot','feedback_state')(io,max_age)
    io.control_seed=lambda max_age=.25:method('ros_io.py','RosRobot','control_seed')(io,max_age)
    io.arm_command=lambda:list(io._arm_cmd)
    return io

def test_missing_gripper_does_not_fabricate_measurement_or_block_five_joints():
    io=io_fixture();assert io.control_seed()==[90.,90.,90.,10.,90.,30.]
    assert io.feedback_state()['positions'][5] is None
    assert io.feedback_state()['valid']==[True]*5+[False]

def test_missing_motion_joint_stale_feedback_and_unknown_command_are_rejected():
    io=io_fixture();io._servo_fb[0][2]=math.nan;assert io.control_seed() is None
    io=io_fixture();io._servo_fb=(io._servo_fb[0],time.monotonic()-1.);assert io.control_seed() is None
    io=io_fixture();io._gripper_command=None;assert io.control_seed() is None

def test_existing_mirror_can_enable_with_measured_first_five_and_saved_gripper():
    io=io_fixture();cfg=config.load();arm=ArmModel(cfg['arm']['tcp_offset_m'])
    mp=mirror.MirrorMapper(arm,cfg['arm'],dict(mirror.load(),relative=True),io.arm_command())
    mp.last_msg=time.monotonic()
    c=SimpleNamespace(mode='manual',busy_arm=False,io=io,mirror=mp,log=lambda s:None)
    method('web/server.py','Console','action')(c,{'action':'mirror','on':True})
    assert c.mode=='mirror' and mp.servo[5]==30.
    assert not hasattr(mp,'begin_alignment')

def test_published_gripper_command_is_saved_without_a_second_publish(tmp_path):
    out=[]
    io=SimpleNamespace(dry_run=False,_gripper_command=30.,_boot_id='test-boot',
      _gripper_path=tmp_path/'gripper.json',arm_pub=SimpleNamespace(publish=out.append),log=lambda s:None)
    method('ros_io.py','RosRobot','_publish_arm')(io,[90,90,90,10,90,36],110)
    assert len(out)==1 and out[0].joint6==36
    assert json.loads(io._gripper_path.read_text())=={'command':36.,'boot_id':'test-boot'}
