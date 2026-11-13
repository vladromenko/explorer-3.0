"""Homebrain client for the isolated Tank agent. Motor watchdog lives on the robot."""
import json
import math
import threading
import time
import cv2
import httpx
import numpy as np
from .paths import DATA


def config():return json.loads((DATA/'configs/tank-agent.json').read_text())

def client():
    cfg=config()
    return httpx.Client(base_url=f"http://{cfg['host']}:{cfg['port']}",headers={'X-Explorer-Token':cfg['token']},
                        trust_env=False,timeout=.25)


def require_response(response, operation):
    if response.status_code==409:
        raise RuntimeError('Tank · '+operation+': '+str(response.json().get('detail','Управление остановлено')))
    response.raise_for_status()

class TankSource:
    def __init__(self,control,mode='manual'):
        self.control=control
        self.mode=mode
        self.http=client()
        self.lease=None
        self.previous_camera=None
        self.provenance={}
        self.sequence=0
        self.stop_requested=lambda:False

    def start(self,explicit_user_start=False,stop=lambda:False):
        if not explicit_user_start or stop():raise InterruptedError('Explicit live operator required')
        self.stop_requested=stop
        observations=[]
        for _ in range(5):
            a=time.monotonic_ns();r=self.http.get('/health');r.raise_for_status();b=time.monotonic_ns()
            observations.append({'rtt_ns':b-a,'offset_ns':(a+b)//2-r.json()['monotonic_ns']})
        clock=min(observations,key=lambda v:v['rtt_ns'])
        self.provenance={'robot':'tank','profile':'MentorPi_Tank','clock_mapping':clock,
            'clock_mapping_kind':'midpoint_estimate_not_shared_clock','physical_skill_verified':False,
            'state_kind':'previous_transmitted_command_estimate','wheel_feedback':'unavailable',
            'motor_acknowledgement':False,'odometry_integral_excluded':True}
        r=self.http.post('/session',json={'mode':self.mode},timeout=20);require_response(r,'запуск');self.lease=r.json()['lease']
        if stop():raise InterruptedError('Operator stopped during preparation')
        return self.provenance

    def begin_episode(self):pass

    def observe(self):
        if self.stop_requested():raise InterruptedError('STOP before observation')
        try:r=self.http.get('/frame.jpg')
        except httpx.TimeoutException as exc:
            raise RuntimeError('Tank: камера не ответила по Wi-Fi за 0,25 с; управление остановлено') from exc
        r.raise_for_status();received=time.monotonic_ns()
        info=json.loads(r.headers['X-Explorer-Camera'])
        status=json.loads(r.headers['X-Explorer-Observation'])
        bgr=cv2.imdecode(np.frombuffer(r.content,dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:raise RuntimeError('Invalid camera JPEG')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        if (rgb.shape[1],rgb.shape[0])!=(640,400):raise RuntimeError('Unexpected Tank camera geometry')
        observation_time=time.monotonic_ns()
        reused=info['sequence']==self.previous_camera;self.previous_camera=info['sequence']
        image={'image':rgb,'sequence':info['sequence'],'timestamp_ns':received,'reused':reused,
               'source_timestamp_ns':info['source_timestamp_ns'],'source_clock':info['source_clock'],
               'agent_receive_timestamp_ns':info['received_ns'],'homebrain_receive_timestamp_ns':received,
               'timestamp_kind':'available_at_homebrain','age_at_agent_response_ns':status['monotonic_ns']-info['received_ns']}
        return {'observation_timestamp_ns':observation_time,
                'observation.state':status['action'],'images':{'front':image},
                'measured_response_kind':'previous_command_estimate_no_wheel_feedback',
                'battery_voltage':status['battery_voltage'],'battery_source':'RRC packet voltage_mV/1000',
                'battery_agent_received_ns':status['battery_received_ns'],'imu_raw':status['imu'],
                'imu_agent_received_ns':status['imu_received_ns'],'robot_clock':'tank_monotonic_ns'}

    def send(self,requested,observation=None):
        if self.stop_requested():raise InterruptedError('STOP before command')
        if len(requested)!=2 or not all(math.isfinite(v) for v in requested):
            raise ValueError('Tank requires two finite velocity channels')
        if observation is not None and time.monotonic_ns()-observation['observation_timestamp_ns']>750_000_000:
            raise RuntimeError('Tank: observation too old to execute this action')
        index=self.sequence;self.sequence+=1
        a=time.monotonic_ns()
        self.provenance['last_command_attempt']={'sequence':index,'requested':list(requested),'send_start_ns':a,'accepted':'unknown'}
        try:response=self.http.post('/command',json={'lease':self.lease,'sequence':index,'v':requested[0],'w':requested[1]},timeout=.5)
        except httpx.TimeoutException as exc:
            raise RuntimeError('Tank: нет ответа на команду по Wi-Fi; управление остановлено') from exc
        b=time.monotonic_ns();require_response(response,'команда');sent=response.json()
        self.provenance['last_command_attempt']['accepted']=True
        return {'action':sent['action'],'requested_action':list(requested),
                'motion_paused':sent.get('motion_paused',False),'pause_reason':sent.get('pause_reason'),
                'action_send_start_ns':a,'action_send_end_ns':b,'action_acknowledged':False,
                'agent_accepted_command':True,'robot_serial_send_start_ns':sent['send_start_ns'],
                'robot_serial_send_end_ns':sent['send_end_ns'],'motor_targets_rps':sent['motor_targets_rps']}

    def sample(self,index,stop=lambda:False):
        if stop():raise InterruptedError('STOP before observation')
        observation=self.observe()
        if stop():raise InterruptedError('STOP before command')
        action=self.send(self.control.base_action(),observation)
        if action['motion_paused']:self.control.clear_base_inputs()
        return {**observation,**action,'frame_id':index,'teleoperation_phase':'paused' if action['motion_paused'] else 'following'}

    def plan_action(self,provider,observation):
        if not provider.needs_inference():return provider.next(observation)
        # Never extend the previous moving command while waiting for GPU inference.
        self.send([0.,0.])
        done=threading.Event();errors=[];started=time.monotonic_ns()
        def hold():
            try:
                while not done.wait(.1):self.send([0.,0.])
            except Exception as exc:errors.append(exc)
        worker=threading.Thread(target=hold,daemon=True);worker.start()
        try:result=provider.next(observation)
        finally:
            done.set();worker.join(timeout=1)
            self.provenance.setdefault('planning_holds',[]).append({
                'start_ns':started,'end_ns':time.monotonic_ns(),'command':[0.,0.]})
        if worker.is_alive():raise RuntimeError('Tank neutral-command worker did not stop')
        if errors:raise errors[0]
        if self.stop_requested():raise InterruptedError('STOP during policy inference')
        return result

    def close(self):
        errors=[]
        try:
            r=self.http.post('/stop',timeout=8);r.raise_for_status()
        except Exception as exc:errors.append('Tank STOP: '+str(exc))
        finally:self.http.close()
        return errors
