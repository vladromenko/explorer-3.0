"""Read-only SO-101 Leader through pinned LeRobot, with no Follower or camera I/O."""
import fcntl
import json
import math
import termios
import time

from .hardware import JOINTS, calibration_snapshot
from .paths import DATA
from .resources import ResourceLease, assert_hardware_available


class LeaderReader:
    def __init__(self, port='/dev/so101-leader'):
        self.port = port
        self.device = None
        self.lease = None

    def start(self):
        reservation = DATA/'configs/so101-external-owner.json'
        if reservation.exists() and not json.loads(reservation.read_text()).get('temporary_leader_only_allowed'):
            raise PermissionError('Leader оставлен прежнему управлению')
        snapshot = calibration_snapshot()
        assert_hardware_available([self.port])
        self.lease = ResourceLease('so101-leader')
        self.lease.__enter__()
        try:
            from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

            class ReadOnlyLeader(SOLeader):
                def disconnect(self):
                    if self.bus.is_connected:
                        self.bus.disconnect(disable_torque=False)

            self.device = ReadOnlyLeader(SOLeaderTeleopConfig(
                id='leader', port=self.port, calibration_dir=DATA/'calibration',
                use_degrees=True, num_read_retries=0))
            bus = self.device.bus
            # SOLeader.connect/configure normally writes operating mode and torque.
            # Use its standard bus + get_action but validate the existing settings read-only.
            bus.connect(handshake=False)
            fcntl.ioctl(bus.port_handler.ser.fileno(), termios.TIOCEXCL)
            bus._handshake()
            if not self.device.is_calibrated:
                raise ValueError('Leader calibration differs; no automatic writes permitted')
            for joint in JOINTS:
                if bus.read('Operating_Mode', joint, normalize=False, num_retry=0) != 0:
                    raise ValueError('Leader requires existing position mode')
                if bus.read('Torque_Enable', joint, normalize=False, num_retry=0) != 0:
                    raise ValueError('Leader torque is enabled; another controller must release it')
            return {'calibration_sha256': snapshot['files']['leader']['sha256'],
                    'driver': 'lerobot.SOLeader', 'version': '0.6.1', 'writes_to_motors': False}
        except Exception:
            self.close()
            raise

    def sample(self):
        if self.device is None:
            raise RuntimeError('Leader not connected')
        started = time.monotonic_ns()
        action = self.device.get_action()
        finished = time.monotonic_ns()
        values = [float(action[joint+'.pos']) for joint in JOINTS]
        if not all(math.isfinite(v) for v in values) or finished-started > 100_000_000:
            raise ValueError('Invalid or stale Leader reading')
        return {'positions': values, 'read_start_ns': started, 'read_end_ns': finished,
                'units': ['degrees']*5+['percent_calibrated_range']}

    def close(self):
        try:
            if self.device is not None:
                self.device.disconnect()
        finally:
            self.device = None
            if self.lease:
                self.lease.__exit__(None, None, None)
                self.lease = None
