"""M4: the drive identification recovers known parameters from synthetic logs."""
import numpy as np

from explorer3.simlab.sysid_base import fit_channel, simulate


def test_recovers_known_drive_parameters():
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(12):
        u = np.repeat(rng.uniform(-0.5, 0.5, 20), 10)            # piecewise-constant commands
        y = simulate(u, 0.0, 0.05, delay=2, tau=0.12, gain=0.8, deadzone=0.05)
        episodes.append((u, y + rng.normal(0, 0.005, len(y))))
    fit = fit_channel(episodes, channel='yaw')
    assert fit.delay == 2
    assert abs(fit.tau - 0.12) < 0.03
    assert abs(fit.gain - 0.8) < 0.05
    assert abs(fit.deadzone - 0.05) < 0.02
    assert fit.accepted and fit.rmse_holdout < fit.rmse_holdout_nominal
