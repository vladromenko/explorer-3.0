"""Reuse the installed MS200 driver after the stock owner has released it."""
import os
import json
from pathlib import Path
import signal
import subprocess

DRIVER=Path('/home/ubuntu/third_party_ros2/third_party_ws/install/oradar_lidar/lib/oradar_lidar/oradar_scan')
DEVICE=Path('/dev/ldlidar')


def command():
    # Copied parameters from the installed peripherals/ms200_scan.launch.py.
    return [str(DRIVER),'--ros-args','-r','__node:=explorer3_ms200',
            '-p','device_model:=MS200','-p','frame_id:=lidar_frame',
            '-p','scan_topic:=/explorer3/tank/scan_raw','-p','port_name:=/dev/ldlidar',
            '-p','baudrate:=230400','-p','angle_min:=0.0','-p','angle_max:=360.0',
            '-p','range_min:=0.05','-p','range_max:=12.0','-p','clockwise:=false','-p','motor_speed:=15']


class Lidar:
    def __init__(self):self.process=None;self.marker=None

    def start(self,root):
        if not DEVICE.exists() or not DRIVER.is_file():raise RuntimeError('Installed MS200 driver/device unavailable')
        # No competing open of the sensor. This never examines or opens the motor port.
        result=subprocess.run(['lsof','-t',str(DEVICE)],capture_output=True,timeout=2)
        if result.returncode!=1 or result.stdout or result.stderr:
            raise RuntimeError('Lidar is still owned, or ownership could not be checked')
        with (root/'lidar.log').open('ab') as log:
            self.process=subprocess.Popen(command(),stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'ROS_LOCALHOST_ONLY':'1'},start_new_session=True)
        self.marker=root/'lidar-active.json'
        self.marker.write_text(json.dumps({'pid':self.process.pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))

    def close(self):
        if self.process:
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
            self.process=None
        if self.marker:self.marker.unlink(missing_ok=True)
