"""M10/M11: run exported policies on robots — one loop per robot, all in parallel.

A loop reads the robot's sensors through a `Source`, builds the contract observation,
asks the numpy Policy for an action and, only in mode 'operator', hands the scaled
command to a `Sink` (the robot agent, which keeps its own clamps, acceleration limit and
safety stops). In mode 'shadow' nothing is sent. Any exception, deadline overrun streak
or stop request ends the loop with sink.stop().

Source.read() -> dict(features=np.ndarray, image=np.ndarray|None, t=float)
Sink.send(physical_action), Sink.stop(reason)
"""
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import contract
from .artifact import Policy
from .perception import TargetTracker


@dataclass
class LoopStats:
    steps: int = 0
    overruns: int = 0
    latency_ms: list = field(default_factory=list)
    stopped_reason: str = ''

    def summary(self):
        lat = np.asarray(self.latency_ms or [0.0])
        return {'steps': self.steps, 'overruns': self.overruns, 'latency_ms_p50': float(np.percentile(lat, 50)),
                'latency_ms_p99': float(np.percentile(lat, 99)), 'stopped_reason': self.stopped_reason}


class PolicyLoop:
    def __init__(self, policy_dir, source, sink, *, mode='shadow', detector=None, lidar_calibration=None,
                 max_overrun_streak=5):
        if mode not in ('shadow', 'operator'):
            raise ValueError('mode must be shadow or operator')
        self.policy = Policy(policy_dir)
        self.robot = self.policy.robot
        self.hz = contract.ROBOTS[self.robot]['control_hz']
        self.source, self.sink, self.mode = source, sink, mode
        self.detector = detector
        self.lidar_calibration = lidar_calibration or {'yaw_offset_rad': 0.0, 'direction': 1}
        self.tracker = TargetTracker()
        self.prev = np.zeros(len(contract.ROBOTS[self.robot]['action']), np.float32)
        self.stats = LoopStats()
        self.stop_event = threading.Event()
        self.max_overrun_streak = max_overrun_streak
        self.proposals = []

    def step(self):
        sample = self.source.read()
        detection = self.detector.detect(sample['image']) if (self.detector is not None and sample.get('image') is not None) else None
        target = self.tracker.update(detection, 1.0 / self.hz)
        obs = contract.from_real_features(self.robot, sample['features'], target, self.prev, self.lidar_calibration)
        action = self.policy.act(obs)
        physical = contract.scale_action(self.robot, action)
        if self.mode == 'operator':
            self.sink.send(physical)
        self.proposals.append(physical.tolist())
        self.prev = action
        return physical

    def run(self, duration_s=None):
        period = 1.0 / self.hz
        start = next_t = time.perf_counter()
        streak = 0
        try:
            while not self.stop_event.is_set():
                if duration_s is not None and time.perf_counter() - start >= duration_s:
                    self.stats.stopped_reason = 'duration'
                    break
                t0 = time.perf_counter()
                self.step()
                dt = time.perf_counter() - t0
                self.stats.steps += 1
                self.stats.latency_ms.append(dt * 1000)
                if dt > period / 2:
                    self.stats.overruns += 1
                    streak += 1
                    if streak >= self.max_overrun_streak:
                        self.stats.stopped_reason = 'latency_overrun'
                        break
                else:
                    streak = 0
                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))
            else:
                self.stats.stopped_reason = self.stats.stopped_reason or 'stop_requested'
        except Exception as exc:  # any failure ends in a stop, never in silent continuation
            self.stats.stopped_reason = f'error: {type(exc).__name__}: {exc}'
        finally:
            self.sink.stop(self.stats.stopped_reason or 'finished')
        return self.stats


class Fleet:
    """Runs several PolicyLoops in parallel threads; stop_all() stops every robot."""

    def __init__(self, loops):
        self.loops = dict(loops)
        self.threads = {}

    def start(self, duration_s=None):
        for name, loop in self.loops.items():
            t = threading.Thread(target=loop.run, kwargs={'duration_s': duration_s}, name=f'policy-{name}', daemon=True)
            self.threads[name] = t
            t.start()

    def stop_all(self):
        for loop in self.loops.values():
            loop.stop_event.set()

    def join(self, timeout=None):
        for t in self.threads.values():
            t.join(timeout)
        return {name: loop.stats.summary() for name, loop in self.loops.items()}
