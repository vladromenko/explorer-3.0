"""ExecStopPost closes only this session's native driver/camera processes.

It never opens a second serial connection or substitutes a different control chain.
A killed driver is not evidence of a physically stopped motor.
"""
import json,os,signal,time
from pathlib import Path
root=Path(__file__).resolve().parent
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for name in ('native-driver-active.json','camera-active.json','lidar-active.json'):
    marker=root/name
    if marker.exists():
        info=json.loads(marker.read_text())
        if info['boot_id']==boot:
            proc=Path('/proc')/str(info['pid'])
            if proc.exists():
                args=(proc/'cmdline').read_bytes().split(b'\0')
                if name=='lidar-active.json':
                    from lidar import DRIVER
                    expected=str(DRIVER).encode()
                else:expected=str(root/'vendor/camera/bin/aurora930_node').encode() if name=='camera-active.json' else b'native_driver.py'
                if expected in args and (proc/'cwd').resolve()==root:
                    os.kill(info['pid'],signal.SIGTERM if name=='native-driver-active.json' else signal.SIGINT)
                    deadline=time.monotonic()+4
                    while proc.exists() and time.monotonic()<deadline:time.sleep(.05)
                    if proc.exists():
                        os.kill(info['pid'],signal.SIGKILL)
                        (root/'shutdown-unconfirmed.json').write_text(json.dumps({'boot_id':boot,'process':name,'physical_stop_confirmed':False}))
                else:raise RuntimeError('Process identity mismatch; not killed')
        marker.unlink()
