#!/usr/bin/env python3
"""Deploy own inert services, with backups and rollback; never acquire hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from explorer3.paths import ROOT,DATA
from scripts.build_staging_bundles import build

HOST_STAGE='/home/pi/explorer3-recording-r2-staging'
STAGE='/home/ubuntu/explorer3-recording-r2-staging'
REMOTE='/home/ubuntu/explorer3'
SERVICES=('explorer3-tank-agent.service','explorer3-tank-sensors@0.service','explorer3-tank-sensors@1.service')


def ssh(*args,timeout=40):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','tank',shlex.join(args)],capture_output=True,text=True,timeout=timeout,check=True).stdout


def docker(*args):return ssh('docker','exec','-u','ubuntu','MentorPi',*args)


def health():
    import httpx
    cfg=json.loads((DATA/'configs/tank-agent.json').read_text())
    with httpx.Client(headers={'X-Explorer-Token':cfg['token']},trust_env=False,timeout=3) as client:
        response=client.get(f"http://{cfg['host']}:{cfg['port']}/health");response.raise_for_status();return response.json()


def deploy():
    current=health()
    if current['armed'] or current['hardware_open']:raise RuntimeError('Tank session active; deployment refused')
    out=DATA/'validation/recording-r2/staging';build(DATA,out)
    ssh('mkdir','-p',HOST_STAGE);docker('mkdir','-p',STAGE)
    for source,name in ((out/'tank-staging.tar','bundle.tar'),(ROOT/'scripts/install_tank_bundle.py','installer.py')):
        subprocess.run(['scp','-q',str(source),'tank:'+HOST_STAGE+'/'+name],check=True,timeout=30)
        ssh('docker','cp',HOST_STAGE+'/'+name,'MentorPi:'+STAGE+'/'+name)
    docker('tar','-xf',STAGE+'/bundle.tar','-C',STAGE)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    host_helper='/home/pi/explorer3/control/handoff.py'
    host_backup=host_helper+'.before-recording-'+stamp
    ssh('sudo','-n','cp','-p',host_helper,host_backup)
    archive=None
    original_services={unit:ssh('systemctl','show',unit,'-p','ActiveState','--value').strip()=='active' for unit in SERVICES}
    report={'created_at':time.time(),'services_before':original_services,'host_backup':host_backup,'physical_commands':0}
    try:
        # Recheck immediately before changing code; no /session, /command or handoff call.
        current=health()
        if current['armed'] or current['hardware_open']:raise RuntimeError('Tank became active before deployment')
        installed=json.loads(docker('python3',STAGE+'/installer.py','--stage',STAGE))
        archive=installed['archive'];report['archive']=archive
        ssh('docker','cp','MentorPi:'+STAGE+'/payload/handoff.py',HOST_STAGE+'/handoff.py')
        ssh('sudo','-n','install','-m','755',HOST_STAGE+'/handoff.py',host_helper)
        for unit in SERVICES:
            if original_services[unit]:ssh('sudo','-n','systemctl','restart',unit)
        deadline=time.monotonic()+20;actual=None
        while time.monotonic()<deadline and actual is None:
            try:actual=health()
            except Exception:time.sleep(.5)
        if actual is None:raise RuntimeError('Updated agent did not become healthy')
        if actual['armed'] or actual['hardware_open']:raise RuntimeError('Updated agent must be inert')
        expected={name:hashlib.sha256((ROOT/'agents/tank'/name).read_bytes()).hexdigest() for name in actual.get('software',{})}
        if not expected or actual['software']!=expected:raise RuntimeError('Runtime agent source mismatch')
        if actual.get('limits')!=json.loads((ROOT/'configs/tank-limits.json').read_text()):raise RuntimeError('Limits mismatch')
        remote=ssh('sha256sum',host_helper).split()[0]
        if remote!=hashlib.sha256((ROOT/'agents/tank/handoff.py').read_bytes()).hexdigest():raise RuntimeError('Host helper hash mismatch')
        report.update(status='installed_verified_inert',health=actual)
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        if archive:
            docker('python3',STAGE+'/installer.py','--rollback',archive)
        ssh('sudo','-n','cp','-p',host_backup,host_helper)
        for unit in SERVICES:
            if original_services[unit]:ssh('sudo','-n','systemctl','restart',unit)
        report['status']='rolled_back'
        (DATA/'validation/recording-r2/deployment.json').write_text(json.dumps(report,indent=2))
        raise
    (DATA/'validation/recording-r2/deployment.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    if not args.apply:parser.error('Explicit --apply required; installer never drives the robot')
    report=deploy();print(json.dumps({k:report[k] for k in ('status','archive','host_backup','physical_commands')},indent=2))
