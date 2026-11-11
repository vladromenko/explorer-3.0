"""ExecStopPost: neutral only after this agent actually owned the board in this boot."""
import importlib.util,json,os,signal,time
from pathlib import Path
root=Path(__file__).resolve().parent
marker=root/'hardware-active.json'
if marker.exists() and json.loads(marker.read_text()).get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip():
    spec=importlib.util.spec_from_file_location('explorer3_vendor_board',root/'vendor/board_sdk.py')
    sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
    board=sdk.Board(device='/dev/rrc',timeout=.1)
    try:
        board.set_motor_speed([[1,0.],[2,0.],[3,0.],[4,0.]])
        board.port.flush()
    finally:board.port.close()
    marker.unlink()

camera=root/'camera-active.json'
if camera.exists():
    info=json.loads(camera.read_text())
    if info['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        proc=Path('/proc')/str(info['pid'])
        if proc.exists() and (proc/'cmdline').read_bytes().split(b'\0')[0]==str(root/'vendor/camera/bin/aurora930_node').encode():
            os.kill(info['pid'],signal.SIGINT)
            deadline=time.monotonic()+2
            while proc.exists() and time.monotonic()<deadline:time.sleep(.1)
            if proc.exists():os.kill(info['pid'],signal.SIGKILL)
    camera.unlink()
