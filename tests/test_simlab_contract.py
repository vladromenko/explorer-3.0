"""Contract mapping from the real sensor feature vector (main environment, numpy only)."""
import numpy as np
import pytest

pytest.importorskip('httpx')
from explorer3.simlab import contract


def test_real_feature_vector_mapping():
    from explorer3.sensors import CHANNELS
    f = np.zeros(len(CHANNELS), np.float32)
    f[0] = 1.5                                   # lidar bin 0
    f[CHANNELS.index('gyro_z')] = 0.4
    f[CHANNELS.index('accel_x')] = 0.2
    f[CHANNELS.index('imu_valid')] = 1
    vec = contract.from_real_features('tank', f, {'distance': 1, 'bearing': 0, 'yaw': 0, 'visible': 1, 'age_s': 0},
                                      [0, 0], {'yaw_offset_rad': 0.0, 'direction': 1})
    names = contract.names('tank')
    assert vec[names.index('lidar_00')] == pytest.approx(1.5 / 4)
    assert vec[names.index('lidar_01')] == 1.0      # no return -> max range
    assert vec[names.index('gyro_z')] == pytest.approx(0.2)
    assert vec[names.index('accel_x')] == pytest.approx(0.1)


def test_contract_indices_match_sensor_channels():
    from explorer3.sensors import CHANNELS
    assert CHANNELS[contract.REAL_LIDAR][0] == 'lidar_min_m_00'
    assert CHANNELS[contract.REAL_IMU] == ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
    assert CHANNELS[contract.REAL_IMU_VALID] == 'imu_valid'
