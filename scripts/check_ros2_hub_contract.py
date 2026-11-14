"""ROS message serialization tests; fake publishers, no DDS traffic or robots."""
import time
import numpy as np
from rclpy.serialization import serialize_message
from explorer3.ros2_hub import Gateway
from explorer3.fleet_live import ClockAlignment


class Publisher:
    def __init__(self,topic):self.topic=topic;self.messages=[]
    def publish(self,msg):
        assert serialize_message(msg)
        self.messages.append(msg)


class Node:
    def __init__(self):self.publishers={}
    def create_publisher(self,kind,topic,qos):
        pub=Publisher(topic);self.publishers[topic]=pub;return pub


node=Node();gateway=Gateway(node);now=time.monotonic_ns();wall=time.time_ns()
meta={'boot_id':'fixture-only','clock_pair':{'monotonic_ns':now,'ros_time_ns':wall,'use_sim_time':False}}
clock=ClockAlignment();clock.add({'before_ns':now-1_000_000,'after_ns':now,'remote':meta})
base={'source_timestamp_ns':wall,'frame_id':'optical','sequence':1,'received_ns':now,'age_s':0.}
streams={
 'rgb':{**base,'encoding':'rgb8'},'depth':{**base,'encoding':'mono16','unit':'mm'},
 'rgb_info':{**base,'width':4,'height':3,'k':[2.,0.,2.,0.,2.,1.,0.,0.,1.],
             'p':[2.,0.,2.,0.,0.,2.,1.,0.,0.,0.,1.,0.],'d':[],'distortion_model':'plumb_bob'},
 'scan':{**base,'frame_id':'lidar','angle_min':0.,'angle_increment':.1,'range_min':.05,'range_max':12.,'scan_time':.1,'time_increment':.001},
 'imu':{**base,'frame_id':'imu'},'cloud':base.copy(),
 'battery':{'sequence':1,'received_ns':now,'age_s':0.,'voltage':7.2},
 'tf_static':{'sequence':1,'transforms':[{'parent':'base_link','child':'optical','translation':[0.,0.,.2],'quaternion_xyzw':[0.,0.,0.,1.]}]}}
arrays={'rgb':np.zeros((3,4,3),np.uint8),'depth':np.full((3,4),1234,dtype='>u2'),
        'scan':np.array([1.,2.,np.inf,np.nan],np.float32),'imu':np.zeros(6,np.float32),
        'cloud':np.array([[.1,.2,1.]],np.float32)}
packet={'meta':{**meta,'streams':streams},'arrays':arrays,'received_ns':now,'request_elapsed_s':0.}
sent,errors=gateway.publish_packet('tank',packet,clock)
assert not errors,errors
assert set(sent)==set(streams)-{'tf_static'}
depth=node.publishers['/explorer3/robots/tank/depth'].messages[0]
assert depth.encoding=='16UC1'
assert np.frombuffer(bytes(depth.data),'<u2').tolist()==[1234]*12 and depth.is_bigendian==0
scan=node.publishers['/explorer3/robots/tank/scan'].messages[0]
assert scan.header.frame_id=='tank/lidar' and np.isnan(scan.ranges[-1])
imu=node.publishers['/explorer3/robots/tank/imu'].messages[0]
assert imu.orientation_covariance[0]==-1
assert gateway.publish_packet('tank',packet,clock)==([], {})  # no replay of a repeated sample
packet['received_ns']-=2_000_000_000
for info in streams.values():info['sequence']=2
assert gateway.publish_packet('tank',packet,clock)==([], {})  # no stale measurements published
assert not any('cmd_vel' in topic for topic in node.publishers)
print('PASS: ROS serialization, namespaced frames, depth endian, unknown orientation, deduplication, stale rejection; no DDS/motion/training')
