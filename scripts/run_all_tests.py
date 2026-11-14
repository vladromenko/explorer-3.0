"""Run isolated groups, keep logs and machine-readable inventory even on failure."""
import argparse
import datetime
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_groups', ROOT/'conftest.py')
groups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groups)
parser = argparse.ArgumentParser()
parser.add_argument('--group', choices=['all', 'main', 'sim', 'browser'], default='all')
args = parser.parse_args()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
parent = Path(os.environ.get('EXPLORER3_DATA', Path.home()/'.local/share/explorer3'))/'validation'
out = parent/f'tests-{stamp}'
out.mkdir(parents=True, exist_ok=False)
inventory = {g: sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_*.py') if groups.test_group(p)==g) for g in ('main','sim','browser')}
report = {'created_at':stamp, 'test_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for files in inventory.values() for f in files}, 'inventory':inventory, 'files':sum(map(len,inventory.values())), 'groups':{}, 'physical_commands':0}
for group, files in inventory.items():
    if args.group in ('all',group) and files:
        interpreter = ROOT/('.venv-sim' if group=='sim' else '.venv')/'bin/python'
        if group=='browser':
            interpreter = Path(os.environ.get('EXPLORER3_BROWSER_PYTHON',str(ROOT/'.venv-browser/bin/python')))
        xml = out/f'{group}.xml'
        with tempfile.TemporaryDirectory(prefix='isolated-',dir=out) as isolated:
            if group=='sim':
                for asset in ('mujoco_menagerie/robotstudio_so101','mentorpi-t1-cad','yahboom-downloads/m3pro-orin/m3pro_ws/src/M3Pro','yahboom-downloads/m3pro-orin/yahboomcar_ws/src/M3Pro_config/config'):
                    source = parent.parent/'cache/source'/asset
                    shutil.copytree(source,Path(isolated)/'cache/source'/asset)
            env = {**os.environ, 'EXPLORER3_DATA':isolated, 'PYTHONPATH':str(ROOT),
                   'MUJOCO_GL':'egl', 'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1',
                   'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1'}
            start = time.monotonic()
            with (out/f'{group}.log').open('w') as log:
                result = subprocess.run([str(interpreter),'-m','pytest','-q',*files,'--junitxml='+str(xml)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            counts = {'tests':0,'failures':0,'errors':0,'skipped':0}
            if xml.exists():
                for suite in ET.parse(xml).getroot().iter('testsuite'):
                    for key in counts:counts[key]+=int(suite.get(key,0))
            counts['passed'] = counts['tests']-counts['failures']-counts['errors']-counts['skipped']
            report['groups'][group] = {**counts,'exit_code':result.returncode,'seconds':round(time.monotonic()-start,2),'log':str(out/f'{group}.log')}
            print(group,json.dumps(report['groups'][group]),flush=True)
        (parent/f'tests-{stamp}.json').write_text(json.dumps(report,indent=2))
report['passed'] = bool(report['groups']) and all(g['exit_code']==0 for g in report['groups'].values())
(parent/f'tests-{stamp}.json').write_text(json.dumps(report,indent=2))
print('Report:',parent/f'tests-{stamp}.json',flush=True)
raise SystemExit(0 if report['passed'] else 1)
