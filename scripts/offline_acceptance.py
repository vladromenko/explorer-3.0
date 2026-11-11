"""Run under the explorer3-offline systemd unit's cgroup network restrictions."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from explorer3.paths import DATA,ROOT,offline_environment

env=offline_environment()
probe=socket.socket()
probe.settimeout(3)
external=probe.connect_ex(('1.1.1.1',443))
probe.close()
if external==0:
    raise RuntimeError('External network was not blocked; refuse offline PASS')
probe=socket.socket()
probe.settimeout(3)
lan=probe.connect_ex(('192.168.0.168',22))
probe.close()
if lan!=0:
    raise RuntimeError('Local LAN unavailable during offline test')
ui_env={**env,'EXPLORER3_PORT':'8031','EXPLORER3_DATA':str(DATA/'validation/offline-ui')}
with (DATA/'logs/offline-ui.log').open('w') as log:
    process=subprocess.Popen([sys.executable,'-m','explorer3.app'],cwd=ROOT,env=ui_env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+60
        health=None
        while health is None and time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError('Offline UI exited; see offline-ui.log')
            try:
                health=json.load(urllib.request.urlopen('http://127.0.0.1:8031/api/health',timeout=2))
            except OSError:
                time.sleep(.5)
        assert health and not health['armed']
        html=urllib.request.urlopen('http://127.0.0.1:8031/',timeout=10).read().decode()
        assert 'Explorer 3.0' in html
        assert 'cdn.tailwindcss.com' not in html
        (DATA/'validation/offline-ui.html').write_text(html)
        print('Offline UI HTTP PASS; LAN available; external socket blocked',flush=True)
        subprocess.run([sys.executable,'scripts/smoke_act.py'],env=env,cwd=ROOT,check=True)
        result={'external_connect_errno':external,'lan_connect_errno':lan,'ui_http':'PASS',
                'act':'PASS','synthetic':True,'hardware_opened':False,
                'isolation':'systemd cgroup IPAddressDeny=any; allow localhost and 192.168.0.0/24'}
        (DATA/'validation/offline-pass.json').write_text(json.dumps(result,indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
