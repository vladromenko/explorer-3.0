#!/usr/bin/env python3
"""Read-only Pi health, sensor freshness and HTTP RTT. Never sends a command."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from explorer3.paths import DATA

REMOTE=r'''
import json,subprocess,time
from pathlib import Path
stat=lambda:list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
a=stat();time.sleep(.25);b=stat();total=sum(b)-sum(a);idle=(b[3]+b[4])-(a[3]+a[4])
p=subprocess.run(['vcgencmd','get_throttled'],capture_output=True,text=True)
t=Path('/sys/class/thermal/thermal_zone0/temp')
print(json.dumps({'cpu_percent':100*(1-idle/total) if total else None,'load_average':Path('/proc/loadavg').read_text().split()[:3],
 'temperature_c':float(t.read_text())/1000 if t.exists() else None,
 'throttled':int(p.stdout.strip().split('=')[-1],16) if p.returncode==0 else None,
 'process_cpu':subprocess.run(['ps','-eo','pid,pcpu,comm','--sort=-pcpu'],capture_output=True,text=True).stdout.splitlines()[:10]}))
'''


def capture():
    import httpx
    cfg=json.loads((DATA/'configs/tank-agent.json').read_text())
    command=shlex.join(['python3','-c',REMOTE])
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','tank',command],capture_output=True,text=True,timeout=15,check=True)
    result=json.loads(p.stdout);result.update(read_only=True,physical_commands=0,created_at=time.time())
    headers={'X-Explorer-Token':cfg['token']}
    rtt=[]
    with httpx.Client(headers=headers,trust_env=False,timeout=3) as client:
        for _ in range(5):
            before=time.monotonic();response=client.get(f"http://{cfg['host']}:{cfg['port']}/health");response.raise_for_status()
            rtt.append((time.monotonic()-before)*1000)
        result['agent']=response.json()
        sensors=client.get(f"http://{cfg['host']}:8043/health");sensors.raise_for_status()
        result['sensors']=sensors.json()
    result['rtt_p95_ms']=max(rtt) # conservative empirical p95 with only five samples
    last=result['agent'].get('last_write')
    result['serial_write_ms']=(last['send_end_ns']-last['send_start_ns'])/1e6 if last else None
    result['serial_latency_note']='Last recorded write only; no probe command is sent, MCU execution unmeasured'
    result['problems']=[]
    if result['throttled']!=0:result['problems'].append('Проверьте питание и охлаждение Tank')
    if result['rtt_p95_ms']>150:result['problems'].append('Поднесите Tank ближе к Wi-Fi')
    streams=result['sensors'].get('streams',{})
    rgb=streams.get('rgb',{});rate=rgb.get('rate_hz')
    if rate is None:result['problems'].append('Частота камеры пока не измерена наблюдателем')
    elif rate<10:result['problems'].append('Камера медленнее 10 кадров/с: проверьте нагрузку Tank')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=capture();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('cpu_percent','temperature_c','throttled','rtt_p95_ms','problems')},ensure_ascii=False,indent=2))
