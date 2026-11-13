"""Tank-local safety gate. Fresh process is inert; copied vendor SDK opens only on explicit start."""
import asyncio
import hmac
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

if __package__:
    from .native_client import NativeDriver
    from .motion_permission import require_motion_permission, limited_command, bench_authorized
else:
    from native_client import NativeDriver
    from motion_permission import require_motion_permission, limited_command, bench_authorized

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text()) if (ROOT/'config.json').exists() else {}



class Hardware:
    def __init__(self):
        self.lock=threading.RLock()
        self.lifecycle=threading.RLock()
        self.cancel_start=threading.Event()
        self.board=None
        self.camera=None
        self.node=None
        self.ros_thread=None
        self.ros_stop=threading.Event()
        self.armed=False
        self.lease=None
        self.expires=0.
        self.last_sequence=-1
        self.last_action=[0.,0.]
        self.reason='Not started'
        self.image=None
        self.image_info=None
        self.frame_sequence=0
        self.battery=None
        self.battery_time=None
        self.imu=None
        self.imu_time=None
        self.done=threading.Event()
        self.watchdog=threading.Thread(target=self.watch,daemon=True)
        self.watchdog.start()

    def status(self):
        with self.lock:
            return {'robot':'tank','armed':self.armed,'hardware_open':self.board is not None,
                    'bench_mode':bench_authorized(ROOT),
                    'reason':self.reason,'action':self.last_action,'motor_acknowledged':False,
                    'physical_stop_confirmed':False,'wheel_feedback':'unavailable','odometry':'not_used_command_integral_is_not_measurement',
                    'monotonic_ns':time.monotonic_ns(),'camera':self.image_info,
                    'battery_voltage':self.battery,'battery_received_ns':self.battery_time,
                    'imu':self.imu,'imu_received_ns':self.imu_time}

    def stop(self,reason):
        with self.lock:
            self.armed=False
            self.expires=0.
            self.lease=None
            self.reason=reason
            if self.board is not None:
                self.board.stop()
            self.last_action=[0.,0.]

    def watch(self):
        while not self.done.wait(.02):
            with self.lock:
                if self.board is not None:
                    try:
                        native=self.board.status()
                        self.battery=native['battery_voltage']
                        self.battery_time=native['battery_received_ns']
                        self.imu=native['imu']
                        self.imu_time=native['imu_received_ns']
                        if self.armed:
                            now=time.monotonic_ns()
                            if native['latched']:
                                self.stop('Native driver: '+str(native['reason']))
                            elif time.monotonic()>self.expires:
                                self.stop('Command lease expired; explicit new start required')
                            elif self.battery_time is None or now-self.battery_time>2_000_000_000:
                                self.stop('Battery telemetry stale')
                            elif self.battery<6.9:
                                self.stop('Low battery: below existing controller threshold 6.9 V')
                            elif self.image_info is None or now-self.image_info['received_ns']>500_000_000:
                                self.stop('Camera stale')
                    except Exception as exc:
                        self.armed=False
                        self.lease=None
                        self.reason='Hardware error: '+str(exc)
                        try:
                            self.board.stop()
                        except Exception as stop_error:
                            self.reason+='; STOP failed: '+str(stop_error)

    def start_camera(self):
        import cv2
        import numpy as np
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        rclpy.init()
        self.node=Node('explorer3_tank_camera_receiver',namespace='/explorer3/tank')
        self.ros_stop.clear()
        def frame(msg):
            if msg.encoding not in ('rgb8','bgr8'):
                return
            array=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
            bgr=cv2.cvtColor(array,cv2.COLOR_RGB2BGR) if msg.encoding=='rgb8' else array
            received=time.monotonic_ns()
            ok,jpeg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,90])
            if ok:
                with self.lock:
                    self.frame_sequence+=1
                    self.image=jpeg.tobytes()
                    self.image_info={'sequence':self.frame_sequence,'received_ns':received,
                        'source_timestamp_ns':msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec,
                        'source_clock':'tank_ros_realtime','width':msg.width,'height':msg.height,
                        'encoding':'jpeg','measured':True}
        self.subscription=self.node.create_subscription(Image,'/explorer3/tank/front/image_raw',frame,qos_profile_sensor_data)
        def spin():
            while not self.ros_stop.is_set():
                rclpy.spin_once(self.node,timeout_sec=.1)
        self.ros_thread=threading.Thread(target=spin,daemon=True);self.ros_thread.start()
        log=(ROOT/'camera.log').open('ab')
        try:
            self.camera=subprocess.Popen([str(ROOT/'vendor/camera/bin/aurora930_node'),'--ros-args',
                '-r','__ns:=/explorer3/tank/camera','--params-file',str(ROOT/'camera.yaml'),
                '-r','/aurora/rgb/image_raw:=/explorer3/tank/front/image_raw',
                '-r','/explorer3/tank/camera/rgb/image_raw:=/explorer3/tank/front/image_raw'],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (ROOT/'camera-active.json').write_text(json.dumps({'pid':self.camera.pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
        finally:
            log.close()

    def start(self):
        if not self.lifecycle.acquire(blocking=False):
            raise RuntimeError('Another lifecycle operation is in progress')
        try:
            self.cancel_start.clear()
            return self._start()
        finally:
            self.lifecycle.release()

    def _start(self):
        require_motion_permission(ROOT)
        if not (ROOT/'physical-block-authorized.json').exists():
            raise PermissionError('Tank physical block has not been authorized')
        marker=ROOT/'handoff-active.json'
        if not marker.exists() or json.loads(marker.read_text()).get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            raise PermissionError('Factory control must release hardware in the current boot')
        if self.board is not None:
            raise RuntimeError('Hardware session already open; close it first')
        if (ROOT/'hardware-active.json').exists():
            raise RuntimeError('Unfinished hardware cleanup; restart the Explorer agent before retrying')
        try:
            driver=NativeDriver()
            try:driver.start()
            except Exception:
                driver.close();raise
            with self.lock:self.board=driver
            if self.cancel_start.is_set():raise RuntimeError('STOP during native startup')
            (ROOT/'hardware-active.json').write_text(json.dumps({'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
            self.start_camera()
            deadline=time.monotonic()+10
            ready=False
            while time.monotonic()<deadline and not ready and not self.cancel_start.is_set():
                with self.lock:
                    ready=self.image_info is not None and self.battery_time is not None
                if not ready:
                    time.sleep(.05)
            with self.lock:
                if (not ready or self.cancel_start.is_set() or self.battery<6.9
                    or time.monotonic_ns()-self.image_info['received_ns']>500_000_000
                    or time.monotonic_ns()-self.battery_time>2_000_000_000):
                    raise RuntimeError('Fresh camera/battery/serial telemetry required')
                self.lease=uuid.uuid4().hex
                self.last_sequence=-1
                self.expires=time.monotonic()+.5
                self.armed=True
                self.reason=None
                return {'lease':self.lease,**self.status()}
        except Exception:
            self.close()
            raise

    def command(self,lease,sequence,v,w):
        with self.lock:
            now=time.monotonic()
            if not self.armed or lease!=self.lease or now>self.expires:
                raise PermissionError('Inactive or expired lease; no automatic rearm')
            if sequence<=self.last_sequence:
                raise ValueError('Stale command sequence')
            if not all(math.isfinite(value) for value in (v,w)):
                raise ValueError('Nonfinite command')
            v,w=limited_command(ROOT,v,w)
            sent=self.board.command(v,w)
            self.last_action=sent['action'];self.last_sequence=sequence;self.expires=now+.35
            return {**sent,'sequence':sequence,'accepted_by_agent':True,'motor_acknowledged':False}

    def close(self):
        self.cancel_start.set()
        errors=[]
        try:self.stop('Explicit STOP / session closed')
        except Exception as exc:errors.append('Neutral command failed: '+str(exc))
        with self.lifecycle:
            self._close(errors)
        if errors:raise RuntimeError('; '.join(errors))

    def _close(self,errors):
        self.ros_stop.set()
        if self.camera is not None:
            try:
                self.camera.send_signal(signal.SIGINT)
                try:self.camera.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    self.camera.kill();self.camera.wait(timeout=2)
                self.camera=None
                (ROOT/'camera-active.json').unlink(missing_ok=True)
            except Exception as exc:errors.append('Camera close: '+str(exc))
        if self.ros_thread is not None:
            self.ros_thread.join(timeout=2)
            if self.ros_thread.is_alive():errors.append('Camera receiver did not stop')
            else:self.ros_thread=None
        if self.node is not None and self.ros_thread is None:
            import rclpy
            try:
                self.node.destroy_node();self.node=None;rclpy.shutdown()
            except Exception as exc:errors.append('ROS close: '+str(exc))
        with self.lock:
            if self.board is not None:
                try:self.board.close();self.board=None
                except Exception as exc:errors.append('Native driver close: '+str(exc))
            if not errors:(ROOT/'hardware-active.json').unlink(missing_ok=True)
            self.image=None;self.image_info=None;self.battery=None;self.battery_time=None
            self.imu=None;self.imu_time=None


hardware=None
@asynccontextmanager
async def lifespan(app):
    global hardware
    hardware=Hardware()
    (ROOT/'agent.pid').write_text(str(os.getpid()))
    yield
    try:hardware.close()
    finally:hardware.done.set();hardware.watchdog.join(timeout=1)

app=FastAPI(lifespan=lifespan)
@app.middleware('http')
async def guard(request:Request,call_next):
    if request.client.host not in CONFIG.get('allowed_clients',[]) or not hmac.compare_digest(request.headers.get('X-Explorer-Token',''),CONFIG.get('token','missing')):
        return Response(status_code=403)
    return await call_next(request)

class Command(BaseModel):
    lease:str
    sequence:int
    v:float
    w:float

@app.get('/health')
def health():return hardware.status()
@app.get('/frame.jpg')
def image():
    with hardware.lock:
        if hardware.image is None:raise HTTPException(503,'No camera frame')
        return Response(hardware.image,media_type='image/jpeg',headers={'X-Explorer-Camera':json.dumps(hardware.image_info),'X-Explorer-Observation':json.dumps(hardware.status()),'Cache-Control':'no-store'})
@app.post('/session')
def start():
    try:return hardware.start()
    except (PermissionError,RuntimeError) as exc:raise HTTPException(409,str(exc)) from exc
@app.post('/command')
def command(value:Command):
    try:return hardware.command(value.lease,value.sequence,value.v,value.w)
    except (PermissionError,ValueError) as exc:raise HTTPException(409,str(exc)) from exc
@app.post('/stop')
def stop():hardware.close();return hardware.status()
