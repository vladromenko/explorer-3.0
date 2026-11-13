"""Stage A keeps ROSMASTER closed unless the operator-agreed ROSMASTER-only exception exists."""
import json, os, subprocess, sys

CODE = """
import json
from explorer3.paths import DATA
from explorer3 import local_work
(DATA/'configs').mkdir(parents=True,exist_ok=True)
(DATA/'configs/workflow.json').write_text(json.dumps({'stage':'A','hardware_connections':False}))
before={r:local_work.hardware_allowed(r) for r in ('rosmaster','tank','so101')}
(DATA/'configs/rosmaster-hardware-enabled.json').write_text(json.dumps({'enabled':True}))
after={r:local_work.hardware_allowed(r) for r in ('rosmaster','tank','so101')}
(DATA/'configs/tank-hardware-enabled.json').write_text(json.dumps({'enabled':True}))
print(json.dumps({'before':before,'after':after,'tank_flag_in_A':local_work.hardware_allowed('tank')}))
"""


def test_stage_a_gating_and_rosmaster_only_exception(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, '-c', CODE], env={**os.environ, 'EXPLORER3_DATA': str(tmp_path), 'PYTHONPATH': root},
                         capture_output=True, text=True, timeout=60, cwd=root)
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r['before'] == {'rosmaster': False, 'tank': False, 'so101': False}
    assert r['after'] == {'rosmaster': True, 'tank': False, 'so101': False}
    assert r['tank_flag_in_A'] is False   # the exception is ROSMASTER-only


def test_stage_scope_overrides_stale_robot_flags(tmp_path,monkeypatch):
    from explorer3 import local_work
    monkeypatch.setattr(local_work,'DATA',tmp_path)
    cfg=tmp_path/'configs';cfg.mkdir()
    (cfg/'local-only.json').write_text(json.dumps({'enabled':True}))
    for robot in ('tank','rosmaster'):(cfg/f'{robot}-hardware-enabled.json').write_text(json.dumps({'enabled':True}))
    (cfg/'workflow.json').write_text(json.dumps({'stage':'B','allowed_robots':['rosmaster']}))
    assert local_work.workflow_stage()=='B'
    assert local_work.hardware_allowed('rosmaster')
    assert not local_work.hardware_allowed('tank') and not local_work.hardware_allowed('so101')
    assert not local_work.hardware_allowed()
    (cfg/'workflow.json').write_text(json.dumps({'stage':'C','allowed_robots':['tank']}))
    assert not local_work.hardware_allowed('rosmaster') and local_work.hardware_allowed('tank')
