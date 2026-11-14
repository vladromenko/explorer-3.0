#!/usr/bin/env python3
"""Install own observation/agent code only while the agent is inactive; no acquire."""
import ast,hashlib,json,shlex,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from explorer3.paths import DATA,ROOT


def deploy():
    cfg=json.loads((DATA/'configs/rosmaster-agent.json').read_text())
    target=cfg.get('ssh_user','vlad')+'@'+cfg['host'];key=str(Path.home()/'.ssh/rosmaster-collector')
    def ssh(*args):
        return subprocess.run(['ssh','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=5',target,shlex.join(args)],capture_output=True,text=True,check=True,timeout=40).stdout
    def inactive():
        state=ssh('systemctl','--user','show','explorer3-rosmaster-agent.service','--property=ActiveState','--value').strip()
        if state!='inactive':raise RuntimeError('Agent must be inactive; deployment does not stop control')
    inactive()
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());root='/home/vlad/explorer3';stage=root+'/staging-readiness-'+stamp
    ssh('mkdir','-p',stage)
    files={'agent.py':ROOT/'agents/rosmaster/agent.py','sensors.py':ROOT/'agents/tank/sensors.py','runtime.py':ROOT/'agents/rosmaster/runtime.py'}
    for source in files.values():ast.parse(source.read_text(),filename=str(source))
    hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
    for name,source in files.items():subprocess.run(['scp','-q','-i',key,str(source),target+':'+stage+'/'+name],check=True,timeout=30)
    inactive()
    code='''import ast,hashlib,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(sys.argv[1]);stage=Path(sys.argv[2]);hashes=json.loads(sys.argv[3]);backup=root/'archive'/stage.name
backup.mkdir(parents=True,exist_ok=False)
services=['explorer3-sensors@0.service','explorer3-sensors@1.service']
active=[s for s in services if subprocess.run(['systemctl','--user','is-active','--quiet',s]).returncode==0]
for n,h in hashes.items():
 assert hashlib.sha256((stage/n).read_bytes()).hexdigest()==h
 ast.parse((stage/n).read_text(),filename=n)
 shutil.copy2(root/n,backup/n)
legacy=root/'stage-d-authorized.json'
try:
 for s in active:subprocess.run(['systemctl','--user','stop',s],check=True)
 for n in hashes:os.replace(stage/n,root/n)
 if legacy.exists():shutil.move(legacy,backup/legacy.name)
 for s in active:subprocess.run(['systemctl','--user','start',s],check=True)
 for n,h in hashes.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
 cfg=json.loads((root/'config/agent.json').read_text())
 for service in active:
  port=8053+int(service.split('@')[1].split('.')[0]);deadline=time.monotonic()+12;verified=False
  while not verified and time.monotonic()<deadline:
   try:
    request=urllib.request.Request('http://127.0.0.1:'+str(port)+'/health',headers={'X-Explorer-Token':cfg['token']})
    with urllib.request.urlopen(request,timeout=1) as response:meta=json.load(response)
    verified=meta.get('source_sha256')==hashes['sensors.py']
   except OSError:pass
   if not verified:time.sleep(.2)
  if not verified:raise RuntimeError('Observer runtime hash mismatch: '+service)
except BaseException:
 for n in hashes:shutil.copy2(backup/n,root/n)
 if (backup/legacy.name).exists():shutil.copy2(backup/legacy.name,legacy)
 for s in active:subprocess.run(['systemctl','--user','restart',s],check=False)
 raise
print(json.dumps({'backup':str(backup),'hashes':hashes,'restarted':active,'agent_started':False,'motion_commands':0}))
'''
    report=json.loads(ssh('python3','-c',code,root,stage,json.dumps(hashes)))
    destination=DATA/'validation/recording-r5-deployment.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    if sys.argv[1:]!=['--apply']:raise SystemExit('Explicit --apply required')
    deploy()
