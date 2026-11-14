"""M4: identify the mobile-base drive model from real logs.

Model (same as dock_gpu.DockBatch.command): measured body rate y_t responds to the command
u_{t-d} after a dead zone, with first-order lag tau and gain (slip) k:
    y_t = y_{t-1} + (1 - exp(-dt/tau)) * (k * dz(u_{t-d}) - y_{t-1})
Fitted separately for the forward channel (measured forward speed: lidar/odometry) and the
yaw channel (IMU gyro z). Multiple shooting: the simulation restarts from the measured value
every `segment` samples, so a bias in one segment does not poison the whole log.
Hold-out: last 20 % of every episode; parameters are accepted only if hold-out RMSE improves
by more than 10 % over the nominal prior.
"""
import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import least_squares

NOMINAL = {'delay': 2, 'tau': 0.10, 'gain': 0.9, 'deadzone': 0.0}


def simulate(u, y0, dt, delay, tau, gain, deadzone, segment=None, y_meas=None):
    u = np.asarray(u, float)
    y = np.zeros(len(u))
    prev = y0
    alpha = 1 - np.exp(-dt / max(tau, 1e-3))
    for t in range(len(u)):
        if segment and y_meas is not None and t % segment == 0:
            prev = y_meas[t - 1] if t > 0 else y0
        c = u[t - delay] if t - delay >= 0 else 0.0
        c = 0.0 if abs(c) < deadzone else c
        prev = prev + alpha * (gain * c - prev)
        y[t] = prev
    return y


@dataclass
class Fit:
    channel: str
    delay: int
    tau: float
    gain: float
    deadzone: float
    rmse_train: float
    rmse_holdout: float
    rmse_holdout_nominal: float
    accepted: bool
    samples: int

    def as_dict(self):
        return asdict(self)


def _rmse(episodes, dt, params, segment, part):
    """RMSE over the train (first 80 %) or hold-out (last 20 %) part of each episode,
    simulated over the whole episode so command history before the split is respected."""
    err = []
    for u, y in episodes:
        if len(u) < 5:
            continue
        sim = simulate(u, y[0], dt, params['delay'], params['tau'], params['gain'], params['deadzone'], segment, y)
        k = int(len(u) * 0.8)
        err.append((sim - y)[:k] if part == 'train' else (sim - y)[k:])
    e = np.concatenate(err) if err else np.zeros(1)
    return float(np.sqrt(np.mean(e ** 2)))


def fit_channel(episodes, dt=0.05, channel='yaw', segment=20, max_delay=5,
                deadzones=np.round(np.arange(0.0, 0.205, 0.01), 3)):
    """episodes: list of (command array, measured response array) at the control rate.
    Delay and dead zone by grid (non-smooth), tau and gain by least squares on the first 80 %."""
    best = None
    for delay in range(0, max_delay + 1):
        for dz in deadzones:
            def residual(x):
                r = []
                for u, y in episodes:
                    if len(u) < 5:
                        continue
                    k = int(len(u) * 0.8)
                    r.append((simulate(u, y[0], dt, delay, x[0], x[1], dz, segment, y) - y)[:k])
                return np.concatenate(r)
            sol = least_squares(residual, x0=[0.1, 0.9], bounds=([0.02, 0.3], [0.6, 1.5]))
            cost = float(np.sqrt(np.mean(sol.fun ** 2)))
            if best is None or cost < best[0]:
                best = (cost, delay, float(dz), sol.x)
    cost, delay, dz, (tau, gain) = best
    params = {'delay': delay, 'tau': float(tau), 'gain': float(gain), 'deadzone': dz}
    rh = _rmse(episodes, dt, params, segment, 'holdout')
    rn = _rmse(episodes, dt, NOMINAL, segment, 'holdout')
    return Fit(channel, delay, float(tau), float(gain), dz, cost, rh, rn, rh < 0.9 * rn,
               int(sum(len(u) for u, _ in episodes)))


def randomization_ranges(fits, widen=0.3):
    """Domain-randomization ranges around identified values (used by dock_gpu when accepted)."""
    out = {}
    for f in fits:
        if not f.accepted:
            continue
        out[f.channel] = {'delay_steps': [max(0, f.delay - 1), f.delay + 1],
                          'tau': [f.tau * (1 - widen), f.tau * (1 + widen)],
                          'gain': [f.gain * (1 - widen / 2), min(1.2, f.gain * (1 + widen / 2))],
                          'deadzone': [0.0, f.deadzone * (1 + widen) + 1e-3]}
    return out


def save(path, fits, source):
    payload = {'source': source, 'fits': [f.as_dict() for f in fits], 'ranges': randomization_ranges(fits)}
    path.write_text(json.dumps(payload, indent=1))
    return payload
