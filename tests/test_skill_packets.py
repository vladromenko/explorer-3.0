import copy

import numpy as np
import pytest

from explorer3.skill_packets import admission, relative_pose, retarget_pose, validate


def packet_fixture():
    phases = []
    for index, name in enumerate(('approach', 'grasp', 'lift', 'hold')):
        anchor = np.eye(4)
        anchor[:3, 3] = [.25, -.05, .016]
        tcp = anchor.copy()
        tcp[2, 3] += [.09, .002, .12, .12][index]
        phases.append({'name': name, 'reference': 'frozen_at_phase_start',
                       'world_object_at_start': anchor.tolist(), 'samples': [
                           {'time_s': float(index), 'world_tcp': tcp.tolist(),
                            'object_tcp': relative_pose(anchor, tcp).tolist(), 'jaw_opening_m': .032}]})
    return {'schema': 'explorer-skill-packet-v1', 'subskill': 'approach_grasp_lift_hold',
            'units': {'translation': 'm', 'time': 's', 'jaw_opening': 'm'},
            'frame_convention': 'T_a_b_maps_b_to_a;right_handed;phase_start_frozen',
            'donor': {'robot': 'so101', 'behavior_source': 'human', 'domain': 'real',
                      'episodes': [{'id': 'source', 'sha256': 'a' * 64}],
                      'usage': {'granted_by': 'owner', 'reference': 'test-fixture-only', 'allowed': ['adaptation']}},
            'object': {'instance_id': 'fixture-cube-1', 'class': 'cube', 'surface': 'rigid_box',
                       'size_m': [.032] * 3, 'size_sigma_m': [.001] * 3},
            'phases': phases,
            'measurement': {'pose_source': 'calibrated_measurement', 'calibration_sha256': 'b' * 64,
                            'camera_sha256': 'c' * 64, 'position_sigma_m': .001,
                            'orientation_sigma_rad': .02, 'jaw_sigma_m': .001},
            'donor_verification': {'status': 'success', 'lift_m': .1, 'held_s': 1.5,
                                   'method': 'operator_reviewed_video', 'evidence_sha256': 'd' * 64,
                                   'evidence_ref': 'fixture-only'}}


def test_rotation_and_translation_preserve_object_relative_grasp():
    source = np.eye(4)
    source[:3, 3] = [1, 2, 3]
    tcp = source.copy()
    tcp[:3, 3] += [.1, .2, .3]
    target = np.eye(4)
    target[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    target[:3, 3] = [-1, 2, 0]
    actual = retarget_pose(target, relative_pose(source, tcp))
    np.testing.assert_allclose(actual[:3, 3], [-1.2, 2.1, .3])
    np.testing.assert_allclose(actual[:3, :3], target[:3, :3])


def test_lift_uses_frozen_reference_when_object_moves_with_hand():
    anchor = np.eye(4)
    tcp_before, tcp_after = np.eye(4), np.eye(4)
    tcp_after[2, 3] = .12
    a = relative_pose(anchor, tcp_before)
    b = relative_pose(anchor, tcp_after)
    assert b[2, 3] - a[2, 3] == pytest.approx(.12)
    packet = packet_fixture()
    packet['phases'][2]['reference'] = 'updated_with_object'
    with pytest.raises(ValueError, match='frozen reference'):
        validate(packet)


def test_measured_source_is_distinct_from_recipient_skill_and_unknown_uncertainty():
    packet = packet_fixture()
    report = admission(packet, domain='real')
    assert report['eligible'] and not report['recipient_verified'] and not report['recipient_policy_trained']
    packet['measurement']['position_sigma_m'] = None
    report = admission(packet)
    assert not report['eligible'] and 'position_sigma_m_unknown' in report['reasons']
    packet['donor_verification']['status'] = 'unverified'
    assert 'donor_grasp_not_verified' in admission(packet)['reasons']


@pytest.mark.parametrize('mutation', ('timestamp', 'reflected_pose', 'raw_servo_units', 'false_lift', 'sim_as_real'))
def test_invalid_donor_claims_are_rejected(mutation):
    packet = copy.deepcopy(packet_fixture())
    if mutation == 'timestamp':
        packet['phases'][1]['samples'][0]['time_s'] = 0
    elif mutation == 'reflected_pose':
        packet['phases'][0]['samples'][0]['world_tcp'][0][0] = -1
    elif mutation == 'raw_servo_units':
        packet['phases'][0]['samples'][0]['jaw_opening_m'] = 45
    elif mutation == 'false_lift':
        packet['donor_verification']['lift_m'] = 0
    else:
        packet['measurement']['pose_source'] = 'simulation_ground_truth'
    with pytest.raises(ValueError):
        validate(packet)
