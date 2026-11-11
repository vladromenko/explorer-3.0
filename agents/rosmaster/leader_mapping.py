"""Relative Leader clutch: no jump on engagement, no commands after a stale sample."""
import math


class LeaderMapping:
    def __init__(self, signs=(1, 1, 1, 1, 1, 1)):
        if len(signs) != 6 or any(s not in (-1, 1) for s in signs):
            raise ValueError('Six verified axis signs required')
        self.signs = tuple(signs)
        self.anchor = None
        self.previous = None
        self.last_time = None
        self.lower = (3, 3, 3, 3, 3, 30)
        self.upper = (177, 177, 177, 177, 267, 170)

    @staticmethod
    def vector(values):
        if len(values) != 6 or not all(math.isfinite(v) for v in values):
            raise ValueError('Six finite channels required')
        return tuple(float(v) for v in values)

    def release(self):
        self.anchor = self.previous = self.last_time = None

    def update(self, leader, robot, now, *, clutch, sample_age):
        leader, robot = self.vector(leader), self.vector(robot)
        if not math.isfinite(now) or not math.isfinite(sample_age) or not 0 <= sample_age <= .1:
            self.release()
            raise ValueError('Leader sample stale')
        if not clutch:
            self.release()
            return [0.0]*6
        if any(not lo <= q <= hi for q, lo, hi in zip(robot, self.lower, self.upper)):
            self.release()
            raise ValueError('ROSMASTER starting pose outside conservative limits')
        if self.anchor is None:
            self.anchor = (leader, robot)
            self.previous, self.last_time = leader, now
            return [0.0]*6
        dt = now-self.last_time
        if not 0 < dt <= .1 or any(abs(q-p) > max(.5, 360*dt) for q, p in zip(leader, self.previous)):
            self.release()
            raise ValueError('Leader discontinuity; explicit re-engagement required')
        self.previous, self.last_time = leader, now
        scale = (.5, .5, .5, .5, .5, .7)  # last channel: degrees per calibrated Leader percent
        targets = [r+s*k*(q-a) for r, s, k, q, a in zip(self.anchor[1], self.signs, scale, leader, self.anchor[0])]
        targets = [min(hi, max(lo, q)) for q, lo, hi in zip(targets, self.lower, self.upper)]
        # Velocity intent goes through copied Ruckig and collision checks, never straight to UART.
        return [max(-12., min(12., 3*(target-current))) if abs(target-current) > .35 else 0.
                for target, current in zip(targets, robot)]
