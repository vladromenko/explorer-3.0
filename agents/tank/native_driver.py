"""Run the copied factory controller + board node; only project lifecycle/STOP wraps them.

--verify-no-hardware substitutes only serial I/O and is never a physical acceptance result.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'native'))
from serial_guard import CheckedSerial,WriteJournal
from motion_permission import require_motion_permission, limited_command, bench_authorized, neutral_diagnostic_authorized


class CaptureSerial:
    """In-memory transport for the complete factory constructors and ROS command path."""
    packets=[]
    out_waiting=0
    is_open=False
    def __init__(self,*args,**kwargs):pass
    def setPort(self,device):self.device=device
    def open(self):self.is_open=True
    def close(self):self.is_open=False
    def read(self,size=1):time.sleep(.01);return b''
    def write(self,packet):self.packets.append(bytes(packet));return len(packet)


def build(verification=False):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import Twist
    from controller.odom_publisher_node import Controller
    import ros_robot_controller.ros_robot_controller_node as native_board
    import ros_robot_controller.ros_robot_controller_sdk as sdk

    if verification:sdk.serial.Serial=CaptureSerial
    journal=WriteJournal(ROOT/('native-verification-writes.jsonl' if verification else 'native-serial-writes.jsonl'))
    class Board(sdk.Board):
        battery=None
        battery_time=None
        imu=None
        imu_time=None
        def __init__(self):
            self.closing=False;self.receive_error=None
            super().__init__()  # Actual factory serial configuration and receiver.
            if not verification:fcntl.ioctl(self.port.fileno(),0x540C)
            self.port=CheckedSerial(self.port,journal)
        def recv_task(self):
            try:super().recv_task()
            except Exception as exc:
                if not self.closing:self.receive_error=str(exc)
        def get_battery(self):
            value=super().get_battery()
            if value is not None:self.battery=value/1000.;self.battery_time=time.monotonic_ns()
            return value
        def get_imu(self):
            value=super().get_imu()
            if value is not None:self.imu=list(value);self.imu_time=time.monotonic_ns()
            return value
    native_board.Board=Board

    class BoardNode(native_board.RosRobotController):
        def __init__(self):
            self.command_lock=threading.RLock()
            self.condition=threading.Condition(self.command_lock)
            self.last_write=None
            self.expected=None
            self.sequence=0
            self.expires=None
            self.latched=False
            self.reason=None
            self.done=threading.Event()
            self.publisher_stopped=threading.Event()
            super().__init__('ros_robot_controller')  # Includes native motor type, voltage and servo offsets.
            self.guard=threading.Thread(target=self.watch,daemon=True);self.guard.start()
        def pub_callback(self):
            try:super().pub_callback()
            finally:self.publisher_stopped.set()
        def neutral(self,reason):
            with self.command_lock:
                if not self.latched:self.reason=reason
                self.latched=True
                self.board.set_motor_speed([[1,0.],[2,0.],[3,0.],[4,0.]])
                journal.emit({'event':'neutral_written','reason':reason,'physical_stop_confirmed':False})
                self.condition.notify_all()
        def set_motor_state(self,msg):
            with self.command_lock:
                values=[[int(item.id),float(item.rps)] for item in msg.data]
                if self.latched and any(v[1] for v in values):return
                matches=self.expected is not None and len(values)==len(self.expected) and all(a[0]==b[0] and math.isclose(a[1],b[1],rel_tol=1e-6,abs_tol=1e-7) for a,b in zip(values,self.expected))
                if not matches:
                    self.neutral('Unexpected motor publisher or command');return
                if not all(math.isfinite(v[1]) and abs(v[1])<=1.2 for v in values):
                    self.neutral('Invalid native motor command');return
                started=time.monotonic_ns()
                try:super().set_motor_state(msg)  # Unmodified factory callback -> unchanged SDK.
                except Exception as exc:
                    try:self.neutral('Serial command failed: '+str(exc))
                    finally:self.condition.notify_all()
                    raise
                self.sequence+=1
                self.last_write={'sequence':self.sequence,'motor_targets_rps':values,'send_start_ns':started,'send_end_ns':time.monotonic_ns(),'motor_acknowledged':False}
                self.expires=time.monotonic()+.35
                self.expected=None
                self.condition.notify_all()
        def watch(self):
            while not self.done.wait(.02):
                with self.command_lock:
                    reason=None
                    if not self.latched and self.expires is not None:
                        if journal.error:reason='Serial journal failed: '+journal.error
                        elif self.board.receive_error:reason='Serial receiver failed: '+self.board.receive_error
                        elif time.monotonic()>self.expires:reason='Native motor command timeout'
                        elif not verification:
                            bench=bench_authorized(ROOT)
                            if (ROOT/'incident-block.json').exists() and not bench and not neutral_diagnostic_authorized(ROOT):reason='Diagnostic permission expired'
                            elif self.board.battery_time is None or time.monotonic_ns()-self.board.battery_time>2_000_000_000:reason='Battery stale'
                            elif self.board.battery<6.9:reason='Low battery'
                    if reason:
                        try:self.neutral(reason)
                        except Exception as exc:self.reason='Neutral write failed: '+str(exc)

    board=BoardNode()
    # Both original constructors initialize ROS. Reuse this process's existing context
    # for the second node; no drivetrain code or source module is rewritten.
    original_init=rclpy.init
    rclpy.init=lambda *args,**kwargs:None
    try:chassis=Controller('odom_publisher')
    finally:rclpy.init=original_init
    publisher=chassis.motor_pub
    command_context=threading.local()
    class Forward:
        def publish(self,msg):
            with board.command_lock:
                # Native ROS subscriptions must not create a new command lease.
                # Only this process's validated HTTP dispatch may reach the SDK.
                if not getattr(command_context,'active',False):
                    board.neutral('Unsolicited ROS velocity command');return
                board.expected=[[int(item.id),float(item.rps)] for item in msg.data]
            publisher.publish(msg)
    chassis.motor_pub=Forward()
    executor=MultiThreadedExecutor(num_threads=2)
    executor.add_node(board);executor.add_node(chassis)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    dispatch=threading.Lock()

    class Driver:
        closed=False
        def status(self):
            return {'battery_voltage':board.board.battery,'battery_received_ns':board.board.battery_time,
                    'imu':board.board.imu,'imu_received_ns':board.board.imu_time,'latched':board.latched,
                    'reason':board.reason,'last_write':board.last_write,'motor_acknowledged':False,
                    'native_driver':True,'verification_no_hardware':verification}
        def command(self,v,w):
            with dispatch:
                if not all(math.isfinite(x) for x in (v,w)):raise ValueError('Nonfinite command')
                if not verification:v,w=limited_command(ROOT,v,w)
                with board.condition:
                    if board.latched:raise PermissionError('Driver latched: '+str(board.reason))
                    if not verification and (board.board.battery_time is None or time.monotonic_ns()-board.board.battery_time>2_000_000_000 or board.board.battery<6.9):
                        raise PermissionError('Fresh battery above existing threshold required')
                    previous=board.sequence
                    msg=Twist();msg.linear.x=float(v);msg.angular.z=float(w)
                    command_context.active=True
                    try:chassis.app_cmd_vel_callback(msg)  # Actual native clamp, corrections and kinematics.
                    finally:command_context.active=False
                    action=[chassis.linear_x,chassis.angular_z]
                    deadline=time.monotonic()+.2
                    while board.sequence==previous and not board.latched and time.monotonic()<deadline:
                        board.condition.wait(timeout=max(0,deadline-time.monotonic()))
                    if board.sequence==previous or board.latched:
                        board.neutral('No native serial completion');raise RuntimeError('Native command not completed')
                    return {'action':action,**board.last_write,'accepted_by_agent':True}
        def stop(self):
            board.neutral('Explicit STOP')
            return self.status()
        def verify_unsolicited(self):
            if not verification:raise PermissionError('In-memory verification only')
            msg=Twist();msg.linear.x=.1
            chassis.app_cmd_vel_callback(msg)
        def close(self):
            if self.closed:return
            self.closed=True
            try:board.neutral('Native driver closing')
            finally:
                errors=[]
                board.done.set();board.running=False
                board.guard.join(timeout=1)
                if not board.publisher_stopped.wait(1):errors.append('Native telemetry publisher did not stop')
                try:
                    executor.shutdown(timeout_sec=1)
                    thread.join(timeout=1)
                    if thread.is_alive():errors.append('Native ROS executor did not stop')
                finally:
                    board.board.closing=True
                    try:board.board.port.close()
                    finally:journal.close()
                if not errors:
                    chassis.destroy_node();board.destroy_node()
                    if rclpy.ok():rclpy.shutdown()
                if errors:raise RuntimeError('; '.join(errors))
    return Driver()


def make_app(driver,close_on_shutdown=False):
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel
    @asynccontextmanager
    async def lifespan(app):
        try:yield
        finally:
            if close_on_shutdown:driver.close()
    app=FastAPI(lifespan=lifespan)
    class Command(BaseModel):v:float;w:float
    @app.get('/health')
    def health():return driver.status()
    @app.post('/command')
    def command(value:Command):
        try:return driver.command(value.v,value.w)
        except (ValueError,PermissionError,RuntimeError) as exc:raise HTTPException(409,str(exc)) from exc
        except OSError as exc:raise HTTPException(503,'Motor serial link failed: '+str(exc)) from exc
    @app.post('/stop')
    def stop():
        try:return driver.stop()
        except OSError as exc:raise HTTPException(503,'STOP not delivered: motor serial link failed. '+str(exc)) from exc
    return app


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-no-hardware',action='store_true')
    parser.add_argument('--verification-case',choices=('timeout','stop','unsolicited'),default='timeout')
    args,ros_args=parser.parse_known_args()
    sys.argv=[sys.argv[0],*ros_args]
    os.environ['MACHINE_TYPE']='MentorPi_Tank'
    if not args.verify_no_hardware:
        require_motion_permission(ROOT)
        marker=json.loads((ROOT/'handoff-active.json').read_text())
        if marker['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise PermissionError('Current-boot handoff required')
        if not (ROOT/'physical-block-authorized.json').exists():raise PermissionError('Explicit physical authorization required')
    driver=build(args.verify_no_hardware)
    if args.verify_no_hardware:
        from fastapi.testclient import TestClient
        try:
            results=[]
            with TestClient(make_app(driver)) as api:
                for v,w in ((.1,0.),(0.,.3),(.1,0.)):
                    response=api.post('/command',json={'v':v,'w':w});response.raise_for_status();results.append(response.json())
                if args.verification_case=='stop':api.post('/stop').raise_for_status()
                elif args.verification_case=='unsolicited':driver.verify_unsolicited()
                else:time.sleep(.5)
                assert driver.status()['latched']
                motor_packets=[p for p in CaptureSerial.packets if len(p)==27 and p[:6]==b'\xaa\x55\x03\x16\x01\x04']
                assert motor_packets and all(struct.unpack_from('<f',motor_packets[-1],7+5*i)[0]==0. for i in range(4))
                stopped_packets=len(CaptureSerial.packets)
                assert api.post('/command',json={'v':.1,'w':0.}).status_code==409
                assert len(CaptureSerial.packets)==stopped_packets
            result={'verification_no_hardware':True,'physical_acceptance':False,'case':args.verification_case,
                    'commands':results,'latched':True,'neutral_packet_verified':True,
                    'rejected_command_wrote_nothing':True,'reason':driver.status()['reason'],
                    'packets':[p.hex() for p in CaptureSerial.packets]}
            (ROOT/('native-verification-'+args.verification_case+'.json')).write_text(json.dumps(result,indent=2))
            print(json.dumps(result))
        finally:driver.close()
    else:
        import uvicorn
        # Uvicorn re-raises SIGTERM after lifespan shutdown; a Python finally alone
        # is not sufficient to stop the ROS executor and flush the serial journal.
        try:uvicorn.run(make_app(driver,close_on_shutdown=True),uds=str(ROOT/'native.sock'),access_log=False,timeout_graceful_shutdown=1)
        finally:driver.close()

if __name__=='__main__':main()
