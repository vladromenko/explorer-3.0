"""Read-only Tank startup evidence. No publishers, serial opens, service changes or buzzer writes."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time


def battery_summary(samples):
    values=[float(v)/1000 for v in samples if isinstance(v,(float,int)) and math.isfinite(v) and 0<v<20000]
    if not values:return {'status':'unavailable','valid_samples':0,'invalid_samples':len(samples)}
    return {'status':'low_voltage_observed' if min(values)<6.9 else 'voltage_above_configured_threshold',
            'min_v':min(values),'max_v':max(values),'valid_samples':len(values),
            'invalid_samples':len(samples)-len(values),'configured_threshold_v':6.9,
            'threshold_source':'previously verified T1 native send_machine_type, 0x1af4; not modified',
            'buzzer_cause_confirmed':False}


def ros_observe(seconds):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import UInt16
    from ros_robot_controller_msgs.msg import BuzzerState
    rclpy.init()
    node=rclpy.create_node('explorer3_read_only_power_probe')
    samples=[];buzzers=[];start=time.monotonic()
    qos=QoSProfile(depth=100,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(UInt16,'/ros_robot_controller/battery',
        lambda m:samples.append({'after_s':round(time.monotonic()-start,3),'millivolts':int(m.data)}),qos)
    node.create_subscription(BuzzerState,'/ros_robot_controller/set_buzzer',
        lambda m:buzzers.append({'after_s':round(time.monotonic()-start,3),'freq':m.freq,
                                'on_time':m.on_time,'off_time':m.off_time,'repeat':m.repeat}),qos)
    try:
        while time.monotonic()-start<seconds:rclpy.spin_once(node,timeout_sec=.1)
        return {'battery':battery_summary([s['millivolts'] for s in samples]),'battery_samples':samples,
                'buzzer_commands':buzzers,'battery_publishers':node.count_publishers('/ros_robot_controller/battery'),
                'buzzer_publishers':node.count_publishers('/ros_robot_controller/set_buzzer'),
                'duration_s':time.monotonic()-start,'read_only':True,
                'limitation':'No ROS buzzer command does not rule out an autonomous board alarm; startup before SSH is not captured'}
    finally:node.destroy_node();rclpy.shutdown()


def run(args,timeout=6,input=None):
    try:
        p=subprocess.run(args,input=input,capture_output=True,text=True,timeout=timeout)
        return {'returncode':p.returncode,'stdout':p.stdout[-40000:],'stderr':p.stderr[-3000:]}
    except subprocess.TimeoutExpired:return {'error':'timeout','seconds':timeout}


def text_file(path):
    try:return Path(path).read_text().replace('\0','').strip()
    except OSError:return None


def host_snapshot():
    usb=[]
    for path in Path('/sys/bus/usb/devices').glob('*'):
        if (path/'idVendor').exists():
            usb.append({k:text_file(path/k) for k in ('idVendor','idProduct','product','manufacturer','serial')})
    native_source_probe = '''import json,hashlib
from pathlib import Path
root=Path('/home/ubuntu/ros2_ws/src')
result=[]
for base in (root/'driver/ros_robot_controller/ros_robot_controller',root/'ros_robot_controller/ros_robot_controller',root/'bringup/bringup'):
 for name in ('ros_robot_controller_node.py','ros_robot_controller_sdk.py','startup_check.py'):
  p=base/name
  if p.is_file():
   data=p.read_bytes();lines=data.decode(errors='replace').splitlines()
   result.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),
      'power_and_buzzer_lines':[str(i+1)+': '+s for i,s in enumerate(lines) if any(k in s for k in ('battery_level','send_machine_type','set_buzzer','msg.freq','msg.repeat'))]})
print(json.dumps(result))'''
    return {'boot_id':text_file('/proc/sys/kernel/random/boot_id'),'uptime':text_file('/proc/uptime'),
            'model':text_file('/proc/device-tree/model'),'meminfo':text_file('/proc/meminfo'),
            'usb':usb,'throttled':run(['vcgencmd','get_throttled']),
            'services':run(['systemctl','is-active','start_node.service','wifi.service','docker.service']),
            'kernel':run(['journalctl','-b','-k','--no-pager','-n','250']),
            'serial_devices':sorted(str(p) for p in Path('/dev').glob('ttyACM*')),
            'serial_owners':run(['sudo','-n','fuser','/dev/ttyACM0']),
            'containers':run(['docker','ps','--format','{{.Names}} {{.Status}}']),
            'native_power_sources':run(['docker','exec','MentorPi','python3','-c',native_source_probe]),
            'read_only':True,'motion_commands_sent':0}


def observe_host(source,seconds):
    print(json.dumps({'kind':'host','data':host_snapshot()}),flush=True)
    command=['docker','exec','-i','MentorPi','bash','-lc',
             'source /opt/ros/humble/setup.bash && source /home/ubuntu/ros2_ws/install/setup.bash && exec python3 - --ros --seconds '+str(seconds)]
    result=run(command,timeout=seconds+12,input=source)
    if result.get('returncode')==0:
        try:result['observation']=json.loads(result['stdout'].strip().splitlines()[-1])
        except (ValueError,IndexError):result['parse_error']=True
    print(json.dumps({'kind':'power','data':result}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ros',action='store_true');parser.add_argument('--seconds',type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.seconds<=30:parser.error('seconds must be 1..30')
    if args.ros:print(json.dumps(ros_observe(args.seconds)),flush=True)
    else:raise SystemExit('Host entry is called by Homebrain watcher with the source attached')


if __name__=='__main__':main()
