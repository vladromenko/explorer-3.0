"""Run against the private copied CubeBot snapshot; no ROS node is constructed."""
from agents.rosmaster.control_adapter import NativeControls
from cubebot import config
from cubebot.arm_model import ArmModel
from cubebot.mirror import load
import pytest


class IO:
    def __init__(self):
        self._arm_cmd=[90.,90.,90.,90.,90.,30.]
        self.feedback=list(self._arm_cmd)
        self.sent=[]
    def arm_command(self):return self._arm_cmd
    def measured_arm(self,**kwargs):return self.feedback
    def drive(self,*args,**kwargs):self.sent.append(('base',args))
    def stop_base(self):self.sent.append(('stop',))
    def move_arm(self,cmd,**kwargs):self.sent.append(('arm',cmd));self._arm_cmd=list(cmd)


def create():
    cfg=config.load();io=IO()
    return io,NativeControls(io,cfg,ArmModel(cfg['arm']['tcp_offset_m']),load())


def test_constructor_inert_and_base_does_not_require_arm_feedback():
    io,c=create();io.feedback=None
    assert not io.sent
    c.enable('rosmaster_dock',0)
    c.set_input({'drive':[1,0,0]},.1);c.tick(.05,.1)
    assert io.sent[-1]==('base',(.3,0.,0.))
    with pytest.raises(PermissionError):c.set_input({'arm':[1,0,0]},.2)
    c.tick(.05,.6)
    assert io.sent[-1]==('stop',) and not c.armed
    with pytest.raises(PermissionError):c.set_input({'drive':[1,0,0]},.7)


def test_arm_cannot_invent_feedback_or_move_base():
    io,c=create();io.feedback=None
    with pytest.raises(ValueError):c.enable('rosmaster_pick_place',0)
    io.feedback=io.arm_command();c.enable('rosmaster_pick_place',0)
    with pytest.raises(PermissionError):c.set_input({'drive':[1,0,0]},.1)


def test_copied_methods_match_recorded_hashes():
    import ast,hashlib,json,inspect
    from pathlib import Path
    import agents.rosmaster.control_adapter as module
    source=Path(module.__file__).read_text();lines=source.splitlines(keepends=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef))
    expected=json.loads((Path(module.__file__).parent/'control-adapter-provenance.json').read_text())
    for name,digest in expected['methods'].items():
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert hashlib.sha256(''.join(lines[method.lineno-1:method.end_lineno]).encode()).hexdigest()==digest
