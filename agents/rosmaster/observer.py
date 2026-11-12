"""Read-only ROS telemetry. Subscribes only; never constructs a serial link or publisher."""
import threading
import time
import math
import numpy as np
import cv2


class Observer:
    def __init__(self):
        self.lock=threading.RLock()
        self.done=threading.Event()
        self.ready=threading.Event()
        self.error=None
        self.thread=None
        self.arm=None;self.arm_ns=None
        self.battery=None;self.battery_ns=None
        self.rgb=None;self.camera_ns=None;self.camera_source_ns=None;self.camera_sequence=0
        self.odom=None;self.odom_ns=None
        self.counts={'arm':0,'battery':0,'camera':0,'odom':0}

    def start(self):
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
        if not self.ready.wait(8) or self.error:raise RuntimeError(self.error or 'ROS observer startup timed out')

    def _run(self):
        node=None;context=None;executor=None
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState,Image
            from nav_msgs.msg import Odometry
            from std_msgs.msg import Float32
            context=Context();rclpy.init(args=[],context=context)
            node=Node('observer',namespace='/explorer3/rosmaster',context=context)
            executor=SingleThreadedExecutor(context=context)
            executor.add_node(node)
            node.create_subscription(JointState,'/arm_servo_state',self._arm,qos_profile_sensor_data)
            node.create_subscription(Float32,'/battery',self._battery,qos_profile_sensor_data)
            node.create_subscription(Image,'/camera/color/image_raw',self._image,qos_profile_sensor_data)
            node.create_subscription(Odometry,'/odom_raw',self._odom,qos_profile_sensor_data)
            self.ready.set()
            while not self.done.is_set():executor.spin_once(timeout_sec=.05)
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            self.ready.set()
        finally:
            if executor is not None:executor.shutdown()
            if node is not None:node.destroy_node()
            if context is not None and context.ok():context.shutdown()

    def _arm(self,msg):
        values=[float(v) if math.isfinite(v) else None for v in msg.position]
        if len(values)==6:
            with self.lock:self.arm=values;self.arm_ns=time.monotonic_ns();self.counts['arm']+=1

    def _battery(self,msg):
        if math.isfinite(msg.data) and 0<msg.data<20:
            with self.lock:self.battery=float(msg.data);self.battery_ns=time.monotonic_ns();self.counts['battery']+=1

    def _odom(self,msg):
        values=[msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.angular.z]
        if all(math.isfinite(v) for v in values):
            with self.lock:self.odom=values;self.odom_ns=time.monotonic_ns();self.counts['odom']+=1

    def _image(self,msg):
        if msg.encoding not in {'rgb8','bgr8'}:raise ValueError('Unsupported RGB encoding: '+msg.encoding)
        if msg.height<=0 or msg.width<=0 or msg.step<msg.width*3:raise ValueError('Invalid camera stride')
        pixels=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
        rgb=pixels[:,:,::-1].copy() if msg.encoding=='bgr8' else pixels.copy()
        with self.lock:
            self.rgb=rgb;self.camera_ns=time.monotonic_ns();self.camera_sequence+=1;self.counts['camera']+=1
            self.camera_source_ns=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec

    def snapshot(self):
        with self.lock:
            now=time.monotonic_ns()
            fresh=lambda stamp:stamp is not None and 0<=now-stamp<500_000_000
            limits=[(0,180),(0,180),(0,180),(0,180),(0,270),(30,180)]
            arm=self.arm if fresh(self.arm_ns) else None
            outside=[i+1 for i,(value,(lo,hi)) in enumerate(zip(arm or [],limits)) if value is not None and not lo<=value<=hi]
            return {'monotonic_ns':now,'clock':'rosmaster_monotonic_ns','read_only':True,
                'motor_commands_sent':False,'error':self.error,'counts':dict(self.counts),
                'arm_measured':arm,'arm_received_ns':self.arm_ns,'arm_outside_native_model':outside,
                'arm_complete':arm is not None and all(v is not None for v in arm),
                'battery_voltage':self.battery if fresh(self.battery_ns) else None,'battery_received_ns':self.battery_ns,
                'measured_velocity':self.odom if fresh(self.odom_ns) else None,'odom_received_ns':self.odom_ns,
                'camera':{'available':fresh(self.camera_ns),'shape':list(self.rgb.shape) if self.rgb is not None else None,
                          'received_ns':self.camera_ns,'source_timestamp_ns':self.camera_source_ns,
                          'source_clock':'ros_header_time_not_homebrain','sequence':self.camera_sequence},
                'controller_command_readiness':'not_measured_by_this_observer','physical_acceptance':False}

    def jpeg(self):
        with self.lock:
            state=self.snapshot()
            if self.error:raise RuntimeError(self.error)
            if not state['camera']['available']:raise RuntimeError('Camera has no fresh image')
            rgb=self.rgb.copy()
        ok,data=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:raise RuntimeError('JPEG encode failed')
        return data.tobytes(),state

    def close(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=2)
        if self.thread and self.thread.is_alive():raise RuntimeError('Observer did not close')


if __name__=='__main__':
    import json
    observer=Observer()
    try:
        observer.start();time.sleep(6)
        print(json.dumps(observer.snapshot()))
    finally:observer.close()
