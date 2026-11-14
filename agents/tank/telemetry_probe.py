"""Finite subscriber-only inspection of factory ROS streams; never opens hardware."""
import json
import math
import os
import time


def timing(samples):
    gaps=[b-a for a,b in zip(samples,samples[1:])]
    return {'samples':len(samples),
            'hz':(len(samples)-1)/(samples[-1]-samples[0]) if gaps else None,
            'max_gap_s':max(gaps) if gaps else None}


def inspect(seconds=15):
    if os.environ.get('ROS_DISTRO')!='humble':
        raise RuntimeError('Tank probe requires ROS Humble')
    os.environ['ROS_DOMAIN_ID']='0'
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import Image,CameraInfo,Imu,LaserScan
    from std_msgs.msg import UInt16
    from geometry_msgs.msg import Twist
    rclpy.init()
    node=rclpy.create_node('explorer3_tank_read_only_telemetry')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    stamps={};last={};battery=[];nonzero=0

    def receive(name,msg):
        nonlocal nonzero
        stamps.setdefault(name,[]).append(time.monotonic())
        if name=='camera_info':
            last[name]={'width':msg.width,'height':msg.height,'k':list(msg.k),'p':list(msg.p),
                        'd':list(msg.d),'distortion_model':msg.distortion_model,'frame_id':msg.header.frame_id}
        elif name=='image':
            last[name]={'width':msg.width,'height':msg.height,'encoding':msg.encoding,'frame_id':msg.header.frame_id}
        elif name=='battery':battery.append(int(msg.data));last[name]={'voltage':msg.data/1000.}
        elif name=='imu':
            last[name]={'angular_velocity_rad_s':[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z],
                        'linear_acceleration_m_s2':[msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z],
                        'frame_id':msg.header.frame_id}
        elif name=='lidar':
            last[name]={'ranges':len(msg.ranges),'finite_ranges':sum(math.isfinite(x) for x in msg.ranges),
                        'range_min':msg.range_min,'range_max':msg.range_max,'frame_id':msg.header.frame_id}
        else:
            values=[msg.linear.x,msg.linear.y,msg.angular.z]
            nonzero+=any(abs(v)>1e-6 for v in values)
            last[name]={'command_not_measurement':values}

    topics={'image':(Image,'/ascamera/camera_publisher/rgb0/image'),
            'camera_info':(CameraInfo,'/ascamera/camera_publisher/rgb0/camera_info'),
            'imu':(Imu,'/ros_robot_controller/imu_raw'),
            'battery':(UInt16,'/ros_robot_controller/battery'),
            'lidar':(LaserScan,'/scan_raw'),
            'velocity_command':(Twist,'/controller/cmd_vel')}
    for name,(kind,topic) in topics.items():
        node.create_subscription(kind,topic,lambda msg,key=name:receive(key,msg),qos)
    start=time.monotonic()
    try:
        while time.monotonic()-start<seconds:rclpy.spin_once(node,timeout_sec=.1)
        return {'read_only':True,'motion_commands_sent':0,'duration_s':time.monotonic()-start,
                'streams':{name:{**timing(stamps.get(name,[])), 'last':last.get(name),
                                 'publishers':node.count_publishers(topic)} for name,(_,topic) in topics.items()},
                'battery_range_v':[min(battery)/1000,max(battery)/1000] if battery else None,
                'nonzero_commands_observed':nonzero,
                'wheel_feedback':'unavailable_in_verified_sdk',
                'odometry':'command_integral_excluded_from_measurements'}
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':print(json.dumps(inspect(),allow_nan=False))
