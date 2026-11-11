"""Stage B read-only motor/calibration probe; no torque or position writes."""
import fcntl,json,termios,time
from dataclasses import asdict
from explorer3.paths import DATA
from explorer3.hardware import JOINTS,hardware_settings,hardware_paths,calibration_snapshot,CameraCapture
from explorer3.resources import ResourceLease,assert_hardware_available
from lerobot.motors import Motor,MotorCalibration,MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
cfg=hardware_settings()
cal=calibration_snapshot()
report={'motor_writes':False,'roles':{}}
with ResourceLease('hardware'),ResourceLease('cameras'):
    assert_hardware_available(hardware_paths())
    for role in ('leader','follower'):
        motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES) for i,n in enumerate(JOINTS,1)}
        values={n:MotorCalibration(**v) for n,v in cal['values'][role].items()}
        bus=FeetechMotorsBus(port=cfg[role+'_port'],motors=motors,calibration=values)
        try:
            bus.connect(handshake=False)
            fcntl.ioctl(bus.port_handler.ser.fileno(),termios.TIOCEXCL)
            bus._handshake()
            actual=bus.read_calibration()
            mismatch=[n for n in JOINTS if any(getattr(actual[n],k)!=getattr(values[n],k) for k in ('id','homing_offset','range_min','range_max'))]
            modes={n:bus.read('Operating_Mode',n,normalize=False,num_retry=0) for n in JOINTS}
            report['roles'][role]={'calibration_matches':not mismatch,'mismatches':mismatch,
              'operating_modes':modes,'raw_position':bus.sync_read('Present_Position',normalize=False,num_retry=1),
              'position':bus.sync_read('Present_Position',num_retry=1),
              'torque':{n:bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in JOINTS}}
            if mismatch or any(v!=0 for v in modes.values()):
                raise RuntimeError(f'{role}: incompatible calibration/mode')
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
    camera=CameraCapture('wrist',cfg['cameras']['wrist'],cfg)
    try:
        camera.start()
        time.sleep(1)
        frame=camera.latest_before(time.monotonic_ns())
        report['camera']={**camera.info,'shape':list(frame['image'].shape),'sequence':frame['sequence']}
        import cv2
        cv2.imwrite(str(DATA/'validation/wrist-stage-b.jpg'),cv2.cvtColor(frame['image'],cv2.COLOR_RGB2BGR))
    finally:
        camera.close()
(DATA/'validation/so101-readonly.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
