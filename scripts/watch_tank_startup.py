"""Collect startup evidence without control writes; only diagnostic files are created on Tank."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('EXPLORER3_DATA',str(Path.home()/'.local/share/explorer3')))
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=2','-o','ConnectionAttempts=1',
     '-o','ServerAliveInterval=2','-o','ServerAliveCountMax=1','hiwonder-tank']


def source_command(source,seconds):
    # Data stays shell-quoted; the only code is the reviewed local probe source.
    code="source="+repr(source)+";ns={'__name__':'tank_probe'};exec(source,ns);ns['observe_host'](source,"+str(seconds)+")"
    return 'python3 -c '+shlex.quote(code)


def detached_command(source,remote):
    worker=("source="+repr(source)+";ns={'__name__':'tank_probe'};exec(source,ns)\n"
            "for i in range(3):\n ns['observe_host'](source,10)\n ns['time'].sleep(3)\n"
            "print('{\"kind\":\"complete\"}',flush=True)\n")
    launcher=("import os,subprocess\nfrom pathlib import Path\n"
              "p=Path("+repr(remote)+");p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)\n"
              "fd=os.open(str(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\n"
              "with os.fdopen(fd,'w') as out:\n"
              " subprocess.Popen(['timeout','120','python3','-u','-c',"+repr(worker)+"],"
              "stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True,close_fds=True)\n")
    return 'python3 -c '+shlex.quote(launcher)


def scan_events(output):
    power_seen=False;complete=False
    for line in output.splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if not isinstance(event,dict):continue
        power_seen |= event.get('data',{}).get('observation',{}).get('battery',{}).get('valid_samples',0)>0
        complete |= event.get('kind')=='complete'
    return power_seen,complete


def write_status(path,**values):
    values['updated_utc']=datetime.now(timezone.utc).isoformat()
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-hours',type=float,default=0)
    p.add_argument('--address',help='Known Tank IP when mDNS is unavailable; keeps SSH alias credentials')
    args=p.parse_args()
    ssh=SSH
    if args.address:
        import ipaddress
        address=str(ipaddress.ip_address(args.address))
        ssh=SSH[:-1]+['-o','Hostname='+address,SSH[-1]]
    if not 0<=args.wait_hours<=72:p.error('wait-hours must be 0 (until connected) or up to 72')
    out=DATA/'validation/corrective-c';out.mkdir(parents=True,exist_ok=True)
    status=out/'startup-watcher.json';end=time.monotonic()+args.wait_hours*3600
    write_status(status,phase='waiting',read_only=True,motion_commands_sent=0)
    while not args.wait_hours or time.monotonic()<end:
        try:
            r=subprocess.run(ssh+['cat /proc/sys/kernel/random/boot_id'],capture_output=True,text=True,timeout=6)
            if r.returncode==0 and len(r.stdout.strip())==36:break
        except subprocess.TimeoutExpired:pass
        time.sleep(10)
    else:write_status(status,phase='expired',read_only=True);return
    boot=r.stdout.strip();folder=out/('startup-'+boot);folder.mkdir(exist_ok=True)
    source=(ROOT/'agents/tank/startup_probe.py').read_text()
    write_status(status,phase='collecting',boot_id=boot,folder=str(folder),read_only=True,motion_commands_sent=0)
    # Launch once on the robot. The bounded reader survives a lost SSH connection;
    # persistent diagnostic files can be retrieved after power/network recovery.
    job_file=folder/'remote-job.json'
    if job_file.exists():
        remote=json.loads(job_file.read_text())['path']
    else:
        remote='/home/pi/.local/state/explorer3/diagnostics/startup-'+str(uuid.uuid4())+'.jsonl'
        job_file.write_text(json.dumps({'path':remote})+'\n')
        try:
            r=subprocess.run(ssh+[detached_command(source,remote)],capture_output=True,text=True,timeout=8)
            (folder/'launch.stderr.txt').write_text(r.stderr)
        except subprocess.TimeoutExpired:
            pass  # Retrieval below can safely retry the exclusive launch.
    power_seen=False
    complete=False
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        try:
            r=subprocess.run(ssh+['cat -- '+shlex.quote(remote)],capture_output=True,text=True,timeout=6)
            if r.returncode==0:
                (folder/'detached.jsonl').write_text(r.stdout)
                power_seen,complete=scan_events(r.stdout)
                if complete:break
            elif r.returncode==1:
                # SSH reached the host but the file does not exist yet. Exclusive
                # creation makes a repeated launch harmless after uncertain delivery.
                subprocess.run(ssh+[detached_command(source,remote)],capture_output=True,text=True,timeout=8)
        except subprocess.TimeoutExpired:pass
        time.sleep(3)
    write_status(status,phase=('captured' if power_seen else 'captured_without_voltage') if complete else 'collection_incomplete',boot_id=boot,folder=str(folder),
                 voltage_observed=power_seen,collection_complete=complete,remote_log=remote,read_only=True,motion_commands_sent=0)


if __name__=='__main__':main()
