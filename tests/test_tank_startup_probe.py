import importlib.util
from pathlib import Path
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(path):
    spec=importlib.util.spec_from_file_location('probe',ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_voltage_unknown_is_never_a_healthy_battery():
    m=module('agents/tank/startup_probe.py')
    assert m.battery_summary([])['status']=='unavailable'
    assert m.battery_summary([0,float('nan'),65535])['status']=='unavailable'
    assert m.battery_summary([6800,7000])['status']=='low_voltage_observed'
    assert m.battery_summary([7900,7950])['min_v']==7.9
    assert not m.battery_summary([6800])['buzzer_cause_confirmed']


def test_probe_does_not_execute_reference_text_as_shell():
    import shlex
    m=module('scripts/watch_tank_startup.py')
    source="value='$(touch NEVER)'; # quotes ` command\n"
    command=shlex.split(m.source_command(source,10))
    assert command[:2]==['python3','-c'] and len(command)==3
    compile(command[2],'<probe>','exec')


def test_ros_probe_is_subscriber_only():
    import ast
    for filename in ('startup_probe.py','telemetry_probe.py'):
        tree=ast.parse((ROOT/'agents/tank'/filename).read_text())
        methods=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        assert 'create_subscription' in methods
        assert not set(methods)&{'create_publisher','publish','Serial','set_buzzer','set_motor_speed'}


@pytest.mark.parametrize('samples',[0,4])
def test_watcher_collects_once_and_distinguishes_missing_voltage(tmp_path,monkeypatch,samples):
    from types import SimpleNamespace
    m=module('scripts/watch_tank_startup.py');m.DATA=tmp_path
    monkeypatch.setattr('sys.argv',['watch_tank_startup.py'])
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    calls=[]
    def ssh(args,**kwargs):
        calls.append(args)
        if args[-1]=='cat /proc/sys/kernel/random/boot_id':
            return SimpleNamespace(returncode=0,stdout='12345678-1234-1234-1234-123456789abc\n',stderr='')
        if args[-1].startswith('python3 -c '):
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        payload={'kind':'power','data':{'observation':{'battery':{'valid_samples':samples}}}}
        return SimpleNamespace(returncode=0,stdout=json.dumps(payload)+'\n'+json.dumps({'kind':'complete'})+'\n',stderr='')
    monkeypatch.setattr(m.subprocess,'run',ssh)
    m.main()
    state=json.loads((tmp_path/'validation/corrective-c/startup-watcher.json').read_text())
    assert state['phase']==('captured' if samples else 'captured_without_voltage')
    assert state['motion_commands_sent']==0 and state['read_only']
    assert len(calls)==3
    assert (Path(state['folder'])/'detached.jsonl').exists()
    assert state['collection_complete']


def test_detached_reader_survives_launcher_and_refuses_duplicate(tmp_path):
    import subprocess,time,shlex
    m=module('scripts/watch_tank_startup.py')
    source="""import time
class Clock:
    def sleep(self,seconds):pass
time=Clock()
def observe_host(source,seconds):
    print('{"kind":"power","data":{"observation":{"battery":{"valid_samples":1}}}}',flush=True)
"""
    path=tmp_path/'with space'/'probe.jsonl'
    command=shlex.split(m.detached_command(source,str(path)))
    subprocess.run(command,check=True,timeout=3)
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        data=path.read_text()
        if m.scan_events(data)==(True,True):break
        time.sleep(.02)
    assert m.scan_events(data)==(True,True)
    before=path.read_bytes()
    retry=subprocess.run(command,capture_output=True,timeout=3)
    assert retry.returncode!=0
    assert path.read_bytes()==before


def test_incomplete_capture_is_not_complete_or_healthy():
    m=module('scripts/watch_tank_startup.py')
    assert m.scan_events('partial output\n{}\n[]\n')==(False,False)
