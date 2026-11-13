"""Wait for the next Tank Wi-Fi connection and collect read-only evidence; no robot changes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('EXPLORER3_DATA',str(Path.home()/'.local/share/explorer3')))
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=2','-o','ConnectionAttempts=1',
     '-o','ServerAliveInterval=2','-o','ServerAliveCountMax=1','hiwonder-tank']


def source_command(source,seconds):
    # Data stays shell-quoted; the only code is the reviewed local probe source.
    code="source="+repr(source)+";ns={'__name__':'tank_probe'};exec(source,ns);ns['observe_host'](source,"+str(seconds)+")"
    return 'python3 -c '+shlex.quote(code)


def write_status(path,**values):
    values['updated_utc']=datetime.now(timezone.utc).isoformat()
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-hours',type=float,default=0);args=p.parse_args()
    if not 0<=args.wait_hours<=72:p.error('wait-hours must be 0 (until connected) or up to 72')
    out=DATA/'validation/corrective-c';out.mkdir(parents=True,exist_ok=True)
    status=out/'startup-watcher.json';end=time.monotonic()+args.wait_hours*3600
    write_status(status,phase='waiting',read_only=True,motion_commands_sent=0)
    while not args.wait_hours or time.monotonic()<end:
        try:
            r=subprocess.run(SSH+['cat /proc/sys/kernel/random/boot_id'],capture_output=True,text=True,timeout=6)
            if r.returncode==0 and len(r.stdout.strip())==36:break
        except subprocess.TimeoutExpired:pass
        time.sleep(3)
    else:write_status(status,phase='expired',read_only=True);return
    boot=r.stdout.strip();folder=out/('startup-'+boot);folder.mkdir(exist_ok=True)
    source=(ROOT/'agents/tank/startup_probe.py').read_text()
    write_status(status,phase='collecting',boot_id=boot,folder=str(folder),read_only=True,motion_commands_sent=0)
    # Multiple windows cover factory ROS startup; none stops/starts services or touches serial.
    power_seen=False
    for index in range(3):
        try:
            r=subprocess.run(SSH+[source_command(source,10)],capture_output=True,text=True,timeout=65)
            (folder/f'probe-{index}.jsonl').write_text(r.stdout)
            (folder/f'probe-{index}.stderr.txt').write_text(r.stderr)
            for line in r.stdout.splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                battery=event.get('data',{}).get('observation',{}).get('battery',{})
                power_seen|=battery.get('valid_samples',0)>0
        except subprocess.TimeoutExpired as exc:
            partial=exc.stdout or b''
            (folder/f'probe-{index}.partial.txt').write_text(partial.decode(errors='replace') if isinstance(partial,bytes) else partial)
        if index<2:time.sleep(3)
    write_status(status,phase='captured' if power_seen else 'captured_without_voltage',boot_id=boot,folder=str(folder),
                 voltage_observed=power_seen,read_only=True,motion_commands_sent=0)


if __name__=='__main__':main()
