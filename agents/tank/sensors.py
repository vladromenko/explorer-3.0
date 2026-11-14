"""Authenticated ROS sensor observer. No publishers, hardware opens or motion routes.

Factory topics and isolated Explorer camera topics are observed; missing streams
remain absent. Raw depth is preserved in millimetres as emitted by Aurora930.
"""
import hmac
import hashlib
from collections import deque
import io
import json
import os
from pathlib import Path
import threading
import time

import numpy as np


def require_ros_environment(robot):
    expected={'tank':('humble','0'),'rosmaster':('jazzy','30')}
    distro,domain=expected[robot]
    if os.environ.get('ROS_DISTRO')!=distro:
        raise RuntimeError(f'{robot} observer requires ROS {distro}; refusing cross-distribution DDS discovery')
    # Set explicitly before ROS initialization, even if caller forgot the environment script.
    os.environ['ROS_DOMAIN_ID']=domain


class SensorStore:
    def __init__(self,robot='tank'):
        if robot not in ('tank','rosmaster'):raise ValueError('Unsupported robot')
        self.robot=robot
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.ros_clock_reader=None
        self.use_sim_time=False
        self.lock=threading.Lock();self.entries={};self.sequence=0
        self.history={name:deque(maxlen=400 if name=='imu' else 60) for name in ('imu','scan','scan_rear')}
        self.boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def put(self,name,array,metadata):
        with self.lock:
            self.sequence+=1
            now=time.monotonic_ns();previous=self.entries.get(name)
            old=previous[1] if previous else {}
            interval=(now-old['received_ns'])/1e9 if previous else None
            rate=1/interval if interval and interval>0 else None
            if rate is not None and old.get('rate_hz') is not None:rate=.2*rate+.8*old['rate_hz']
            self.entries[name]=(array,{**metadata,'sequence':self.sequence,'received_ns':now,
                'frame_count':old.get('frame_count',0)+1,'rate_hz':rate})
            if name in self.history and array is not None:
                self.history[name].append((np.asarray(array).copy(),now,metadata.get('source_timestamp_ns',0)))

    def capture(self,include_history=True):
        with self.lock:
            entries=dict(self.entries)
            histories={name:list(values) for name,values in self.history.items()} if include_history else {}
        now=time.monotonic_ns()
        ros_now=self.ros_clock_reader() if self.ros_clock_reader else None
        clock_after=time.monotonic_ns()
        meta={'version':1,'robot':self.robot,'source_sha256':self.source_sha256,'boot_id':self.boot_id,'monotonic_ns':now,
              'clock_pair':{'monotonic_ns':(now+clock_after)//2,'ros_time_ns':ros_now,
                            'uncertainty_ns':(clock_after-now)//2,'system_time_ns':time.time_ns(),
                            'use_sim_time':self.use_sim_time},
              'read_only':True,'motion_commands_sent':0,'streams':{}}
        arrays={}
        for key,(value,metadata) in entries.items():
            meta['streams'][key]={**metadata,'age_s':(now-metadata['received_ns'])/1e9}
            if value is not None:arrays[key]=value
        for name,values in histories.items():
            if values:
                shape=values[-1][0].shape
                compatible=[item for item in values if item[0].shape==shape]
                arrays[name+'_history']=np.stack([item[0] for item in compatible])
                arrays[name+'_history_received_ns']=np.array([item[1] for item in compatible],dtype=np.int64)
                arrays[name+'_history_source_ns']=np.array([item[2] for item in compatible],dtype=np.int64)
                meta.setdefault('history',{})[name]={'samples':len(compatible),
                    'kind':'all_received_samples_in_bounded_window_not_all_published',
                    'clock':'robot_monotonic_ns','source_clock':entries[name][1].get('source_clock')}
        return meta,arrays


def decode_image(msg):
    channels={'rgb8':3,'bgr8':3,'mono8':1,'mono16':1,'16UC1':1,'32FC1':1}.get(msg.encoding)
    if channels is None:raise ValueError('Unsupported image encoding: '+msg.encoding)
    dtype=np.dtype(('>' if msg.is_bigendian else '<')+('u2' if msg.encoding in ('mono16','16UC1') else 'f4' if msg.encoding=='32FC1' else 'u1'))
    if msg.step<msg.width*channels*dtype.itemsize or len(msg.data)!=msg.height*msg.step:
        raise ValueError('Invalid image row stride')
    pixels=np.ndarray((msg.height,msg.width,channels),dtype=dtype,buffer=bytes(msg.data),
                      strides=(msg.step,channels*dtype.itemsize,dtype.itemsize)).copy()
    if channels==1:return pixels[:,:,0]
    return pixels[:,:,::-1].copy() if msg.encoding=='bgr8' else pixels


def observe(store,done):
    require_ros_environment(store.robot)
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import Image,CameraInfo,Imu,LaserScan,PointCloud2,MagneticField,JointState
    from nav_msgs.msg import Odometry
    from std_msgs.msg import UInt16,Float32
    from tf2_msgs.msg import TFMessage
    from rclpy.qos import DurabilityPolicy
    rclpy.init();node=rclpy.create_node('explorer3_sensor_observer')
    store.ros_clock_reader=lambda:node.get_clock().now().nanoseconds
    store.use_sim_time=node.get_parameter('use_sim_time').value
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    def callback(name,topic,msg):
        meta={'topic':topic}
        if hasattr(msg,'header'):
            meta.update(frame_id=msg.header.frame_id,source_timestamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                        source_clock=store.robot+'_ros_time')
        array=None
        if isinstance(msg,Image):
            array=decode_image(msg);meta.update(encoding=msg.encoding,width=msg.width,height=msg.height)
            if name=='depth':meta['unit']='mm' if msg.encoding in ('mono16','16UC1') else 'm'
        elif isinstance(msg,CameraInfo):
            meta.update(width=msg.width,height=msg.height,k=list(msg.k),p=list(msg.p),r=list(msg.r),d=list(msg.d),distortion_model=msg.distortion_model)
        elif isinstance(msg,LaserScan):
            array=np.asarray(msg.ranges,dtype=np.float32)
            meta.update(angle_min=msg.angle_min,angle_increment=msg.angle_increment,range_min=msg.range_min,
                        range_max=msg.range_max,time_increment=msg.time_increment,scan_time=msg.scan_time,unit='m')
        elif isinstance(msg,Imu):
            array=np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z,
                            msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z],dtype=np.float32)
            meta['units']=['rad/s']*3+['m/s2']*3
            meta.update(orientation_xyzw=[msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w],
                        orientation_covariance=list(msg.orientation_covariance),
                        angular_velocity_covariance=list(msg.angular_velocity_covariance),
                        linear_acceleration_covariance=list(msg.linear_acceleration_covariance))
        elif isinstance(msg,UInt16):meta['voltage']=msg.data/1000.
        elif isinstance(msg,Float32):meta['voltage']=float(msg.data)
        elif isinstance(msg,MagneticField):
            array=np.array([msg.magnetic_field.x,msg.magnetic_field.y,msg.magnetic_field.z],np.float32)
            meta.update(unit='tesla',covariance=list(msg.magnetic_field_covariance))
        elif isinstance(msg,JointState):
            array=np.array(msg.position,np.float32);meta.update(names=list(msg.name),unit='servo_degrees_from_CubeBot_bridge')
        elif isinstance(msg,Odometry):
            array=np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,
                msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w,
                msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.angular.z],np.float32)
            meta.update(child_frame=msg.child_frame_id,source='CubeBot measured encoders',
                        pose_z=msg.pose.pose.position.z,pose_covariance=list(msg.pose.covariance),
                        twist_covariance=list(msg.twist.covariance),
                        channels=['x_m','y_m','qx','qy','qz','qw','vx_m_s','vy_m_s','wz_rad_s'])
        elif isinstance(msg,PointCloud2):
            # Display-only decimation of the manufacturer's metric cloud. Full
            # depth + CameraInfo are retained separately for offline reconstruction.
            fields={f.name:f for f in msg.fields}
            if not all(k in fields and fields[k].datatype==7 for k in ('x','y','z')):return
            dtype=np.dtype({'names':['x','y','z'],'formats':[('>' if msg.is_bigendian else '<')+'f4']*3,
                            'offsets':[fields[k].offset for k in ('x','y','z')],'itemsize':msg.point_step})
            points=np.ndarray((msg.height,msg.width),dtype=dtype,buffer=bytes(msg.data),strides=(msg.row_step,msg.point_step))
            stride=max(1,(msg.width*msg.height+5999)//6000)
            array=np.stack([points[k].reshape(-1)[::stride] for k in ('x','y','z')],axis=-1).copy()
            array=array[np.isfinite(array).all(axis=1)&(np.linalg.norm(array,axis=1)>0)]
            meta.update(unit='m',decimation_stride=stride,original_points=msg.width*msg.height,display_only=True)
        store.put(name,array,meta)
    subscriptions={
        'rgb':(Image,['/ascamera/camera_publisher/rgb0/image','/explorer3/tank/front/image_raw']),
        'depth':(Image,['/ascamera/camera_publisher/depth0/image_raw','/explorer3/tank/camera/depth/image_raw']),
        'ir':(Image,['/ascamera/camera_publisher/ir0/image','/explorer3/tank/camera/ir/image_raw']),
        'rgb_info':(CameraInfo,['/ascamera/camera_publisher/rgb0/camera_info','/explorer3/tank/camera/rgb/camera_info']),
        'depth_info':(CameraInfo,['/ascamera/camera_publisher/depth0/camera_info','/explorer3/tank/camera/ir/camera_info']),
        'scan':(LaserScan,['/scan_raw','/explorer3/tank/scan_raw']),
        'imu':(Imu,['/ros_robot_controller/imu_raw','/explorer3/tank/ros_robot_controller/imu_raw']),
        'battery':(UInt16,['/ros_robot_controller/battery','/explorer3/tank/ros_robot_controller/battery']),
        'cloud':(PointCloud2,['/ascamera/camera_publisher/depth0/points','/explorer3/tank/camera/points2'])}
    if store.robot=='rosmaster':
        subscriptions={
            'rgb':(Image,['/camera/color/image_raw']),
            'depth':(Image,['/camera/depth/image_raw']),
            'ir':(Image,['/camera/ir/image_raw']),
            'rgb_info':(CameraInfo,['/camera/color/camera_info']),
            'depth_info':(CameraInfo,['/camera/depth/camera_info']),
            'scan':(LaserScan,['/scan0']),
            'scan_rear':(LaserScan,['/scan1']),
            'imu':(Imu,['/imu/data_raw']),
            'mag':(MagneticField,['/imu/mag']),
            'arm':(JointState,['/arm_servo_state']),
            'odometry':(Odometry,['/odom_raw']),
            'battery':(Float32,['/battery']),
            'cloud':(PointCloud2,['/camera/depth/points','/camera/depth_registered/points'])}
    def safe_callback(name,topic,msg):
        try:callback(name,topic,msg)
        except (ValueError,TypeError) as exc:store.put(name+'_error',None,{'error':str(exc),'topic':topic})
    for name,(kind,topics) in subscriptions.items():
        for topic in topics:node.create_subscription(kind,topic,lambda msg,n=name,t=topic:safe_callback(n,t,msg),qos)
    transforms={}
    def tf_static(msg):
        for t in msg.transforms:
            transforms[t.child_frame_id]={'parent':t.header.frame_id,'child':t.child_frame_id,
                'translation':[t.transform.translation.x,t.transform.translation.y,t.transform.translation.z],
                'quaternion_xyzw':[t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w]}
        store.put('tf_static',None,{'transforms':list(transforms.values())})
    node.create_subscription(TFMessage,'/tf_static',tf_static,QoSProfile(depth=100,
        reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    try:
        while not done.is_set():rclpy.spin_once(node,timeout_sec=.1)
    finally:node.destroy_node();rclpy.shutdown()


def make_app(store,token):
    from fastapi import FastAPI,HTTPException,Request
    from fastapi.responses import Response
    app=FastAPI()
    @app.middleware('http')
    async def authenticate(request:Request,call_next):
        if not token or not hmac.compare_digest(request.headers.get('X-Explorer-Token',''),token):
            return Response(status_code=403)
        return await call_next(request)
    @app.get('/health')
    def health():return store.capture(include_history=False)[0]
    @app.get('/microphone.wav')
    def microphone():
        if __package__:from .microphone import capture
        else:from microphone import capture
        try:return Response(capture(),media_type='audio/wav',headers={'Cache-Control':'no-store'})
        except (RuntimeError,TimeoutError) as exc:raise HTTPException(503,str(exc)) from exc
    @app.get('/snapshot.npz')
    def snapshot():
        meta,arrays=store.capture()
        if not arrays:raise HTTPException(503,'Датчики ещё не передали данные')
        # RGB is already a video modality; compress only colour for Wi-Fi.
        # Depth/IR/scan/IMU remain lossless numeric arrays.
        if 'rgb' in arrays:
            import cv2
            ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(arrays.pop('rgb'),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise HTTPException(503,'RGB encoding failed')
            arrays['rgb_jpeg']=encoded
            meta['streams']['rgb']['transport_encoding']='jpeg_quality_85'
        import zipfile
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for name,value in {'metadata':np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8),**arrays}.items():
                with archive.open(name+'.npy','w') as entry:np.lib.format.write_array(entry,value,allow_pickle=False)
        return Response(output.getvalue(),media_type='application/octet-stream',headers={'Cache-Control':'no-store'})
    return app


if __name__=='__main__':
    import argparse
    import os
    import uvicorn
    parser=argparse.ArgumentParser()
    parser.add_argument('--localhost',choices=('0','1'),default='0')
    parser.add_argument('--robot',choices=('tank','rosmaster'),default='tank')
    parser.add_argument('--port',type=int)
    args=parser.parse_args()
    require_ros_environment(args.robot)
    os.environ['ROS_LOCALHOST_ONLY']=args.localhost
    (Path(__file__).parent/f'sensors-{args.localhost}.pid').write_text(str(os.getpid()))
    root=Path(__file__).parent
    token=json.loads((root/('config.json' if args.robot=='tank' else 'config/agent.json')).read_text())['token']
    store=SensorStore(args.robot);done=threading.Event()
    thread=threading.Thread(target=observe,args=(store,done),daemon=True);thread.start()
    try:uvicorn.run(make_app(store,token),host='0.0.0.0',port=args.port or 8043+int(args.localhost),access_log=False)
    finally:done.set();thread.join(timeout=2)
