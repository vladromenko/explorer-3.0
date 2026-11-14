"""Finite MCAP capture. Subscriptions only; fixed domain and observation allowlist."""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,choices=(15,30,60),required=True)
    args=parser.parse_args()
    if (os.environ.get('ROS_DISTRO'),os.environ.get('ROS_DOMAIN_ID'),os.environ.get('ROS_LOCALHOST_ONLY'))!=('jazzy','73','1'):
        raise RuntimeError('Capture requires isolated Homebrain Jazzy domain 73')
    import rclpy
    from rclpy.serialization import serialize_message,deserialize_message
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
    root=Path('/capture');manifest=json.loads((root/'manifest.json').read_text())
    types={'rgb':'sensor_msgs/msg/Image','depth':'sensor_msgs/msg/Image','ir':'sensor_msgs/msg/Image',
        'rgb_info':'sensor_msgs/msg/CameraInfo','depth_info':'sensor_msgs/msg/CameraInfo',
        'scan':'sensor_msgs/msg/LaserScan','scan_rear':'sensor_msgs/msg/LaserScan',
        'imu':'sensor_msgs/msg/Imu','mag':'sensor_msgs/msg/MagneticField',
        'odometry':'nav_msgs/msg/Odometry','servo_feedback_rad':'sensor_msgs/msg/JointState',
        'battery':'sensor_msgs/msg/BatteryState','cloud_display':'sensor_msgs/msg/PointCloud2'}
    topics={f'/explorer3/robots/{r}/{k}':v for r in ('tank','rosmaster') for k,v in types.items()}
    topics.update({'/explorer3/fleet/status':'std_msgs/msg/String','/tf_static':'tf2_msgs/msg/TFMessage'})
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(root/'bag'),storage_id='mcap',max_bagfile_size=64*1024**2),
                rosbag2_py.ConverterOptions('cdr','cdr'))
    rclpy.init();node=rclpy.create_node('explorer3_observation_capture')
    stats={};total=0;limit=512*1024**2;status_samples=[]
    def receive(topic,msg):
        nonlocal total
        blob=serialize_message(msg)
        if total+len(blob)>limit:return
        now=time.time_ns();writer.write(topic,blob,now);total+=len(blob)
        s=stats.setdefault(topic,{'count':0,'first_received_ns':now,'type':topics[topic]})
        s.update(count=s['count']+1,last_received_ns=now)
        if hasattr(msg,'header'):
            s.update(frame_id=msg.header.frame_id,last_source_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec)
        if topic=='/explorer3/fleet/status' and len(status_samples)<600:status_samples.append(json.loads(msg.data))
    for index,(topic,kind) in enumerate(topics.items(),1):
        writer.create_topic(rosbag2_py.TopicMetadata(id=index,name=topic,type=kind,serialization_format='cdr'))
        qos=QoSProfile(depth=100,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE) if topic=='/tf_static' else qos_profile_sensor_data
        node.create_subscription(get_message(kind),topic,lambda msg,t=topic:receive(t,msg),qos)
    start=time.monotonic()
    try:
        while time.monotonic()-start<args.seconds and total<limit-8*1024**2:rclpy.spin_once(node,timeout_sec=.1)
    finally:
        duration=time.monotonic()-start
        node.destroy_node();rclpy.shutdown();del writer
    # Read the finished MCAP with the official reader: no playback publisher.
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(root/'bag'),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
    from explorer3.fleet_replay import ReplayIndex
    replay=ReplayIndex(root)
    counts={};frames={}
    while reader.has_next():
        topic,blob,stamp=reader.read_next();counts[topic]=counts.get(topic,0)+1
        if any(topic.endswith('/'+k) for k in ('rgb','depth','scan','scan_rear')):
            replay.observe(topic,deserialize_message(blob,get_message(topics[topic])),stamp)
        if topic not in frames:
            msg=deserialize_message(blob,get_message(topics[topic]))
            frames[topic]=getattr(getattr(msg,'header',None),'frame_id',None)
    assert counts=={t:s['count'] for t,s in stats.items()},'MCAP readback count mismatch'
    assert any('/robots/' in t for t in counts),'No robot observations captured'
    del reader
    manifest.update(phase='complete',duration_s=round(duration,3),bytes_serialized=total,
        finished_at=time.time(),streams=stats,readback={'passed':True,'counts':counts,'frames':frames},
        status_samples=len(status_samples),limit_reached=total>=limit-8*1024**2)
    manifest['observed_robots']=[r for r in ('tank','rosmaster') if any(t.startswith(f'/explorer3/robots/{r}/') for t in stats)]
    manifest['matched_clock_samples']=sum(bool(s.get('pairing',{}).get('matched')) for s in status_samples)
    manifest['replay_frames']=replay.finish()
    (root/'clock-status.json').write_text(json.dumps(status_samples,ensure_ascii=False))
    temp=root/'manifest.tmp';temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));temp.replace(root/'manifest.json')
    print('Saved and read back',sum(counts.values()),'messages',flush=True)


if __name__=='__main__':main()
