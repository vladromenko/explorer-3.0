"""Run in the hub image: actual DDS subscriber, no motion endpoints or training."""
import json
import time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,LaserScan,Imu,JointState
from nav_msgs.msg import Odometry

rclpy.init();node=rclpy.create_node('explorer3_hub_acceptance');received={}
types={'rgb':Image,'depth':Image,'scan':LaserScan,'scan_rear':LaserScan,'imu':Imu,
       'servo_feedback_rad':JointState,'odometry':Odometry}
for key,kind in types.items():
    node.create_subscription(kind,'/explorer3/robots/rosmaster/'+key,
                             lambda msg,k=key:received.update({k:msg}),qos_profile_sensor_data)
deadline=time.monotonic()+18
while time.monotonic()<deadline and len(received)<len(types):rclpy.spin_once(node,timeout_sec=.2)
assert set(received)==set(types),set(types)-set(received)
now=time.time();result={}
for name,msg in received.items():
    age=now-msg.header.stamp.sec-msg.header.stamp.nanosec/1e9
    assert -.15<age<3,(name,age)
    assert msg.header.frame_id.startswith('rosmaster/')
    result[name]={'age_s':round(age,4),'frame':msg.header.frame_id}
assert received['depth'].encoding=='16UC1' and received['depth'].step==2*received['depth'].width
assert np.count_nonzero(np.frombuffer(bytes(received['depth'].data),dtype='<u2'))>100
assert len(received['scan'].ranges)>100 and len(received['scan_rear'].ranges)>100
assert received['scan'].header.frame_id!=received['scan_rear'].header.frame_id
assert len(received['servo_feedback_rad'].position)==6
topics=node.get_publisher_names_and_types_by_node('explorer3_observation_hub','/')
assert not any('cmd_vel' in name or name.endswith('/command') for name,_ in topics)
result.update(result='passed',motion_commands_sent=0,training_started=False,topics=[n for n,_ in topics])
print(json.dumps(result,indent=2));node.destroy_node();rclpy.shutdown()
