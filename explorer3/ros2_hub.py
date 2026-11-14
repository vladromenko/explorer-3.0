"""Read-only ROS 2 gateway for authenticated Explorer sensor observations.

Runs in its own Jazzy container/domain. No robot ROS subscriptions, command
publishers, device mounts or acquisition endpoints. Each source has its own poll
thread so an offline robot cannot stall another. Only fresh, new samples publish.
"""
import json
import signal
import threading
import time
from collections import deque
import numpy as np
from .paths import DATA
from .sensors import SensorClient,stream_age
from .fleet_live import ClockAlignment,frame_id,match_observations


class Source:
    def __init__(self,robot):
        self.robot=robot;self.lock=threading.Lock();self.latest=None;self.error=None
        self.clock=ClockAlignment();self.done=threading.Event()
        self.history=deque(maxlen=8)
        self.thread=threading.Thread(target=self.run,daemon=True)

    def run(self):
        clients=[]
        try:
            clients=[SensorClient(robot=self.robot,isolated=False),SensorClient(robot=self.robot,isolated=True)]
            last_clock=0;chosen=None
            while not self.done.is_set():
                start=time.monotonic()
                try:
                    packet=None;used=None
                    for client in clients:
                        try:
                            candidate=client.snapshot()
                            if stream_age(candidate,'rgb')<1.:
                                packet=candidate;used=client;break
                        except Exception:continue
                    if packet is None:raise RuntimeError('Нет свежих потоков датчиков')
                    if time.monotonic()-last_clock>3 or chosen is not used:
                        samples=[used.clock_sample() for _ in range(3)]
                        with self.lock:
                            if chosen is not used:self.clock=ClockAlignment()
                            for sample in samples:self.clock.add(sample)
                        last_clock=time.monotonic();chosen=used
                    with self.lock:
                        estimate=self.clock.estimate(packet['meta'])
                        stamp,kind=self.clock.stamp(packet['meta'],packet['meta']['streams']['rgb'])
                        if kind=='source_header':
                            self.history.append({'timestamp_ns':stamp,'uncertainty_ns':estimate['uncertainty_ns'],
                                'boot_id':packet['meta']['boot_id'],'generation':self.clock.generation,
                                'packet_id':f"{packet['meta']['boot_id']}:{packet['meta']['monotonic_ns']}"})
                        self.latest=packet;self.error=None
                except Exception as exc:
                    with self.lock:self.error=str(exc)
                    last_clock=0
                self.done.wait(max(.05,.5-(time.monotonic()-start)))
        except Exception as exc:
            with self.lock:self.error=str(exc)
        finally:
            for client in clients:client.close()


class Gateway:
    def __init__(self,node):
        from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
        from std_msgs.msg import String
        from tf2_ros import StaticTransformBroadcaster
        self.node=node;self.publishers={};self.seen={};self.tf_seen={}
        self.qos=QoSProfile(depth=2,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.state_pub=node.create_publisher(String,'/explorer3/fleet/status',QoSProfile(depth=1,
            reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.static=StaticTransformBroadcaster(node)

    def emit(self,robot,key,msg):
        topic=f'/explorer3/robots/{robot}/{key}'
        if topic not in self.publishers:self.publishers[topic]=self.node.create_publisher(type(msg),topic,self.qos)
        self.publishers[topic].publish(msg)

    def publish_packet(self,robot,packet,clock):
        from sensor_msgs.msg import Image,CameraInfo,LaserScan,Imu,MagneticField,JointState,PointCloud2,PointField,BatteryState
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TransformStamped
        meta=packet['meta'];arrays=packet['arrays'];streams=meta['streams'];published=[];errors={}
        def header(msg,info):
            stamp,_=clock.stamp(meta,info)
            msg.header.stamp.sec=stamp//10**9;msg.header.stamp.nanosec=stamp%10**9
            msg.header.frame_id=frame_id(robot,info['frame_id'])
        for key,info in streams.items():
            token=(meta['boot_id'],clock.generation,info['sequence'])
            if self.seen.get((robot,key))==token:continue
            if key!='tf_static' and stream_age(packet,key)>1.:continue
            try:
                value=arrays.get(key);msg=None
                if key in ('rgb','depth','ir') and value is not None:
                    msg=Image();header(msg,info)
                    msg.height,msg.width=value.shape[:2]
                    msg.encoding='rgb8' if key=='rgb' else info['encoding']
                    if key=='depth':
                        # Aurora names its millimetre depth mono16. Expose the
                        # standard depth encoding; preserve the source in packets.
                        if info.get('unit')=='mm' and value.dtype.kind=='u' and value.dtype.itemsize==2:
                            msg.encoding='16UC1'
                        elif info.get('unit')=='m' and value.dtype.kind=='f' and value.dtype.itemsize==4:
                            msg.encoding='32FC1'
                        else:raise ValueError('Unsupported depth units or dtype')
                    little=np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<')))
                    msg.is_bigendian=0;msg.step=little.strides[0];msg.data=little.tobytes()
                elif key in ('rgb_info','depth_info'):
                    msg=CameraInfo();header(msg,info);msg.width=info['width'];msg.height=info['height']
                    msg.k=info['k'];msg.p=info['p'];msg.r=info.get('r',[1.,0.,0.,0.,1.,0.,0.,0.,1.])
                    msg.d=info['d'];msg.distortion_model=info['distortion_model']
                elif key in ('scan','scan_rear') and value is not None:
                    msg=LaserScan();header(msg,info)
                    for name in ('angle_min','angle_increment','range_min','range_max','time_increment','scan_time'):
                        setattr(msg,name,float(info[name]))
                    msg.angle_max=msg.angle_min+msg.angle_increment*(len(value)-1);msg.ranges=value.astype(np.float32).tolist()
                elif key=='imu' and value is not None:
                    msg=Imu();header(msg,info)
                    msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z=map(float,value[:3])
                    msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z=map(float,value[3:])
                    q=info.get('orientation_xyzw',[0.,0.,0.,1.]);msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w=q
                    for field in ('orientation_covariance','angular_velocity_covariance','linear_acceleration_covariance'):
                        setattr(msg,field,info.get(field,[-1.]+[0.]*8))
                elif key=='mag' and value is not None:
                    msg=MagneticField();header(msg,info)
                    msg.magnetic_field.x,msg.magnetic_field.y,msg.magnetic_field.z=map(float,value)
                    msg.magnetic_field_covariance=info.get('covariance',[0.]*9)
                elif key=='arm' and value is not None:
                    # Raw feedback angles, NOT calibrated URDF joint coordinates.
                    # Keep this distinction in the topic and diagnostics.
                    msg=JointState();header(msg,{**info,'frame_id':info.get('frame_id') or 'servo_feedback'})
                    msg.name=info['names'] or [f'servo_{i+1}' for i in range(len(value))]
                    msg.position=np.deg2rad(value).astype(float).tolist()
                elif key=='odometry' and value is not None:
                    msg=Odometry();header(msg,info);msg.child_frame_id=frame_id(robot,info['child_frame'])
                    msg.pose.pose.position.x=float(value[0]);msg.pose.pose.position.y=float(value[1]);msg.pose.pose.position.z=info.get('pose_z',0.)
                    q=msg.pose.pose.orientation;q.x,q.y,q.z,q.w=map(float,value[2:6])
                    msg.twist.twist.linear.x=float(value[6]);msg.twist.twist.linear.y=float(value[7]);msg.twist.twist.angular.z=float(value[8])
                    msg.pose.covariance=info.get('pose_covariance',[0.]*36);msg.twist.covariance=info.get('twist_covariance',[0.]*36)
                elif key=='cloud' and value is not None:
                    msg=PointCloud2();header(msg,info);msg.height=1;msg.width=len(value)
                    msg.fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(('x','y','z'))]
                    msg.point_step=12;msg.row_step=12*len(value);msg.is_dense=True
                    msg.data=np.ascontiguousarray(value,dtype='<f4').tobytes()
                elif key=='battery':
                    msg=BatteryState();header(msg,{**info,'frame_id':'battery'})
                    msg.voltage=float(info['voltage']);msg.present=True
                    for field in ('current','charge','capacity','design_capacity','percentage','temperature'):setattr(msg,field,float('nan'))
                elif key=='tf_static':
                    digest=json.dumps(info['transforms'],sort_keys=True)
                    if self.tf_seen.get(robot)!=digest:
                        transforms=[]
                        for source in info['transforms']:
                            t=TransformStamped();t.header.frame_id=frame_id(robot,source['parent']);t.child_frame_id=frame_id(robot,source['child'])
                            t.transform.translation.x,t.transform.translation.y,t.transform.translation.z=source['translation']
                            t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w=source['quaternion_xyzw']
                            transforms.append(t)
                        self.static.sendTransform(transforms);self.tf_seen[robot]=digest
                if msg is not None:self.emit(robot,{'arm':'servo_feedback_rad','cloud':'cloud_display'}.get(key,key),msg);published.append(key)
                self.seen[(robot,key)]=token
            except (ValueError,TypeError,KeyError,AssertionError) as exc:errors[key]=str(exc)
        return published,errors


def main():
    import os
    if os.environ.get('ROS_DISTRO')!='jazzy' or os.environ.get('ROS_DOMAIN_ID')!='73' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('Hub requires Jazzy, isolated domain 73 and localhost-only discovery')
    import rclpy
    from std_msgs.msg import String
    rclpy.init();node=rclpy.create_node('explorer3_observation_hub');gateway=Gateway(node)
    sources={robot:Source(robot) for robot in ('tank','rosmaster')}
    for source in sources.values():source.thread.start()
    done=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:done.set())
    directory=DATA/'ros2-hub';directory.mkdir(exist_ok=True)
    try:
        while not done.is_set():
            status={'version':1,'running':True,'updated_at':time.time(),'domain_id':73,'read_only':True,
                    'motion_commands_sent':0,'shared_map_ready':False,'robots':{}}
            candidates={}
            for robot,source in sources.items():
                with source.lock:
                    packet=source.latest;state={'error':source.error,'published_streams':[],'ages_s':{}}
                    if packet:
                        state['ages_s']={k:round(stream_age(packet,k),3) for k in packet['meta']['streams'] if k!='tf_static'}
                        state['source_stamps_ns']={k:v.get('source_timestamp_ns') for k,v in packet['meta']['streams'].items()}
                        state['source_frames']={k:v.get('frame_id') for k,v in packet['meta']['streams'].items() if v.get('frame_id')}
                        try:
                            estimate=source.clock.estimate(packet['meta'])
                            state.update(uncertainty_ms=round(estimate['uncertainty_ns']/1e6,3),
                                         boot_id=packet['meta']['boot_id'],clock_kind=estimate['kind'])
                            if not source.error:
                                sent,errors=gateway.publish_packet(robot,packet,source.clock)
                                state['published_streams']=sent;state['conversion_errors']=errors
                                now=time.time_ns()
                                candidates[robot]=[{**v,'age_s':(now-v['timestamp_ns'])/1e9} for v in source.history
                                    if v['boot_id']==source.clock.boot and v['generation']==source.clock.generation]
                        except Exception as exc:state['error']=str(exc)
                status['robots'][robot]=state
            status['pairing']=match_observations(candidates)
            text=json.dumps(status,ensure_ascii=False,allow_nan=False)
            temp=directory/'status.tmp';temp.write_text(text);temp.replace(directory/'status.json')
            gateway.state_pub.publish(String(data=text));rclpy.spin_once(node,timeout_sec=.05);done.wait(.15)
    finally:
        for source in sources.values():source.done.set()
        for source in sources.values():source.thread.join(timeout=4)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
