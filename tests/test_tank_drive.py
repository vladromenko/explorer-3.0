"""Tank driving from the Explorer panel: native controller, clamps, TTL zero, no permit files."""
import time

import pytest

from agents.tank import agent, limits, motion_permission


class Board:
    def __init__(self, battery=7.6):
        self.sent, self.battery = [], battery
    def status(self):
        return {'battery_voltage': self.battery, 'battery_received_ns': time.monotonic_ns(), 'imu': None,
                'imu_received_ns': None, 'latched': False, 'reason': None}
    def command(self, v, w):
        self.sent.append([v, w]); return {'action': [v, w], 'send_start_ns': 1, 'send_end_ns': 2, 'motor_targets_rps': []}
    def stop(self):
        self.sent.append('STOP')
    def close(self):
        pass


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'ROOT', tmp_path)
    h = agent.Hardware()
    yield h
    h.done.set(); h.watchdog.join(timeout=1)


def arm(h, board):
    h.board = board; h.armed = True; h.lease = 'op'; h.expires = time.monotonic() + 1


def test_no_permit_files_gate_motion(tmp_path):
    (tmp_path / 'incident-block.json').write_text('{}')          # legacy files are ignored
    assert motion_permission.require_motion_permission(tmp_path) is None
    assert motion_permission.limited_command(tmp_path, .2, .5) == (.2, .5)


def test_speed_clamps_follow_factory_ceiling(tmp_path):
    value = limits.load(tmp_path)
    assert limits.clip(value, 5., -9.) == (.3, -1.)                         # manual defaults
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/tank-limits.json').write_text('{"version":2,"manual":{"v":0.9,"w":3.0}}')
    assert limits.clip(limits.load(tmp_path), 5., -9.) == (.6, -2.)        # never above the factory ceiling


def test_drives_without_camera_lidar_imu(hw):
    arm(hw, Board())
    out = hw.command('op', 1, .2, .5)
    assert out['accepted_by_agent'] and hw.board.sent[-1] != [0., 0.]
    assert not out['motion_paused']


def test_command_pause_sends_zero_but_keeps_session(hw):
    arm(hw, Board())
    for i in range(1, 12):
        hw.command('op', i, .3, 0.)
    hw.expires = time.monotonic() - .01
    deadline = time.monotonic() + 1
    while hw.board.sent[-1] != [0., 0.] and time.monotonic() < deadline:
        time.sleep(.01)
    assert hw.board.sent[-1] == [0., 0.] and hw.armed
    assert hw.command('op', 99, .2, 0.)['accepted_by_agent']      # resumes without a new start


def test_low_but_usable_battery_does_not_block(hw):
    arm(hw, Board(battery=6.6)); hw.battery = 6.6
    assert hw.command('op', 1, .2, 0.)['action'] != [0., 0.]


def test_deep_discharge_and_optional_obstacle_stop():
    value = limits.load('/nonexistent')
    assert limits.guard(value, .2, 0., battery=6.0)[0] == (0., 0.)
    assert limits.guard(value, .2, 0., battery=7.5, front=.05)[0] == (.2, 0.)      # obstacle stop off by default
    value['obstacle_stop'] = True
    assert limits.guard(value, .2, .3, battery=7.5, front=.05)[0] == (0., .3)       # turning in place still allowed
    assert limits.guard(value, -.2, 0., battery=7.5, front=.05)[0] == (-.2, 0.)     # reversing away allowed


def test_stop_always_works(hw):
    arm(hw, Board())
    hw.stop('STOP')
    assert hw.board.sent[-1] == 'STOP' and not hw.armed
