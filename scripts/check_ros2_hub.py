"""Run in the hub image: actual DDS subscribers, no motion endpoints or training."""
import argparse
import json
import os
import time
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--robots',nargs='+',choices=('tank','rosmaster'),default=['rosmaster'])
args=parser.parse_args()
assert os.environ.get('ROS_DISTRO')=='jazzy'
assert os.environ.get('ROS_DOMAIN_ID')=='73' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,LaserScan,Imu,JointState,BatteryState
from nav_msgs.msg import Odometry

rclpy.init();node=rclpy.create_node('explorer3_hub_acceptance');received={}
base={'rgb':Image,'depth':Image,'scan':LaserScan,'imu':Imu,'battery':BatteryState}
contracts={robot:{**base,**({'scan_rear':LaserScan,'servo_feedback_rad':JointState,'odometry':Odometry}
                         if robot=='rosmaster' else {})} for robot in args.robots}
for robot,types in contracts.items():
    for key,kind in types.items():
        node.create_subscription(kind,f'/explorer3/robots/{robot}/{key}',
            lambda msg,r=robot,k=key:received.update({(r,k):msg}),qos_profile_sensor_data)
expected={(r,k) for r,types in contracts.items() for k in types}
deadline=time.monotonic()+25
while time.monotonic()<deadline and set(received)!=expected:rclpy.spin_once(node,timeout_sec=.2)
assert set(received)==expected,expected-set(received)
now=time.time();result={}
for (robot,name),msg in received.items():
    age=now-msg.header.stamp.sec-msg.header.stamp.nanosec/1e9
    assert -.15<age<3,(robot,name,age)
    assert msg.header.frame_id.startswith(robot+'/'),(robot,name,msg.header.frame_id)
    result[f'{robot}/{name}']={'age_s':round(age,4),'frame':msg.header.frame_id}
for robot in contracts:
    depth=received[robot,'depth'];scan=received[robot,'scan']
    assert depth.encoding=='16UC1' and depth.step==2*depth.width
    assert np.count_nonzero(np.frombuffer(bytes(depth.data),dtype='<u2'))>100
    assert len(scan.ranges)>100
    assert 0<received[robot,'battery'].voltage<20
    if robot=='rosmaster':
        rear=received[robot,'scan_rear']
        assert len(rear.ranges)>100 and scan.header.frame_id!=rear.header.frame_id
        assert len(received[robot,'servo_feedback_rad'].position)==6
topics=node.get_publisher_names_and_types_by_node('explorer3_observation_hub','/')
assert not any('cmd_vel' in name or name.endswith('/command') for name,_ in topics)
result.update(result='passed',motion_commands_sent=0,training_started=False,topics=[n for n,_ in topics])
print(json.dumps(result,indent=2));node.destroy_node();rclpy.shutdown()
