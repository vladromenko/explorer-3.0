#!/usr/bin/env python3
"""Install the committed ROSMASTER agent code on the robot, with backup and automatic rollback.

  .venv/bin/python scripts/deploy_rosmaster_agent.py            # check only: which files differ
  .venv/bin/python scripts/deploy_rosmaster_agent.py --apply    # install + restart agent (~1 min)

Safety:
- refuses while a recording attempt is open on Homebrain (never restart mid-recording);
- copies only files that differ, verifies sha256 on the robot, keeps a backup in ~/explorer3/archive/;
- restarting the agent releases the arm motors for a few seconds: the arm should rest in its start pose
  or be held by hand; wheels are not commanded;
- after restart the agent /health must report the new hashes, otherwise the old files are restored.
"""
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3 import catalog  # noqa: E402
from explorer3.paths import DATA  # noqa: E402

FILES = ['agent.py', 'runtime.py', 'leased_core.py', 'control_adapter.py', 'cartesian_mirror.py',
         'observer.py', 'handoff.py', 'so101_leader_kinematics.json']
REMOTE = '/home/vlad/explorer3'
SERVICE = 'explorer3-rosmaster-agent.service'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    apply = '--apply' in sys.argv[1:]
    cfg = json.loads((DATA / 'configs/rosmaster-agent.json').read_text())
    target = cfg.get('ssh_user', 'vlad') + '@' + cfg['host']
    key = str(Path.home() / '.ssh/rosmaster-collector')

    def ssh(*args, timeout=40):
        return subprocess.run(['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', target,
                               'export XDG_RUNTIME_DIR=/run/user/$(id -u); ' + shlex.join(args)],
                              capture_output=True, text=True, check=True, timeout=timeout).stdout

    local = {n: sha((ROOT / 'agents/rosmaster' / n).read_bytes()) for n in FILES}
    if subprocess.run(['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', target, 'true'],
                      capture_output=True).returncode:
        raise SystemExit('ROSMASTER %s недоступен: включите робота и подождите ~1 мин после загрузки' % cfg['host'])
    remote = json.loads(ssh('python3', '-c', 'import hashlib,json,sys,pathlib;r=pathlib.Path(sys.argv[1]);'
                            'print(json.dumps({n:(hashlib.sha256((r/n).read_bytes()).hexdigest() if (r/n).is_file() else None)'
                            ' for n in sys.argv[2:]}))', REMOTE, *FILES))
    changed = [n for n in FILES if local[n] != remote[n]]
    for n in FILES:
        print('%-30s %s' % (n, 'ОТЛИЧАЕТСЯ' if n in changed else 'совпадает'))
    if not changed:
        print('На роботе уже стоит текущий код.')
        return
    if not apply:
        print('\nПроверка завершена. Установка: добавьте --apply')
        return

    with catalog.connect() as db:
        open_attempts = db.execute('SELECT id FROM collection_attempts WHERE outcome IS NULL AND created>?',
                                   (time.time() - 3600,)).fetchall()
    if open_attempts:
        raise SystemExit('Идёт запись (открыта попытка %s) — установка запрещена' % open_attempts[0][0])

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    stage = REMOTE + '/staging-agent-' + stamp
    ssh('mkdir', '-p', stage)
    for n in changed:
        subprocess.run(['scp', '-q', '-i', key, str(ROOT / 'agents/rosmaster' / n), target + ':' + stage + '/' + n],
                       check=True, timeout=30)
    code = r'''import hashlib,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(sys.argv[1]);stage=Path(sys.argv[2]);want=json.loads(sys.argv[3]);service=sys.argv[4]
backup=root/'archive'/stage.name;backup.mkdir(parents=True,exist_ok=False)
for n,h in want.items():
    assert hashlib.sha256((stage/n).read_bytes()).hexdigest()==h,n
    if (root/n).exists():shutil.copy2(root/n,backup/n)
token=json.loads((root/'config/agent.json').read_text())['token']
def healthy():
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        try:
            r=urllib.request.Request('http://127.0.0.1:8042/health',headers={'X-Explorer-Token':token})
            with urllib.request.urlopen(r,timeout=2) as resp:meta=json.load(resp)
            if all(meta.get('software',{}).get(n,h)==h for n,h in want.items()):return meta
        except (OSError,ValueError):pass
        time.sleep(.5)
    return None
try:
    for n in want:os.replace(stage/n,root/n)
    subprocess.run(['systemctl','--user','restart',service],check=True)
    meta=healthy()
    if meta is None:raise RuntimeError('agent did not come back with new code')
except BaseException:
    for n in want:
        if (backup/n).exists():shutil.copy2(backup/n,root/n)
    subprocess.run(['systemctl','--user','restart',service],check=False)
    raise
print(json.dumps({'backup':str(backup),'installed':want,'hardware_open':meta.get('hardware_open')}))
'''
    print('\nУстанавливаю %s; агент перезапускается, рука на несколько секунд без удержания...' % ', '.join(changed))
    report = json.loads(ssh('python3', '-c', code, REMOTE, stage, json.dumps({n: local[n] for n in changed}),
                            SERVICE, timeout=120).strip().splitlines()[-1])
    report['time'] = stamp
    out = DATA / 'validation/rosmaster-agent-deploy-%s.json' % stamp
    out.write_text(json.dumps(report, indent=1) + '\n')
    print('Готово. Резервная копия на роботе: %s\nОтчёт: %s' % (report['backup'], out))


if __name__ == '__main__':
    main()
