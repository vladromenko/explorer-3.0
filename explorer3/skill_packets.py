"""Versioned donor artifacts carried by the existing knowledge store, never motor commands.

T_a_b maps homogeneous column coordinates from b into a; metres, right-handed.
Each phase freezes its object reference. In particular, lifting the grasped object
must not move the reference along with the TCP and erase the lift displacement.
MimicGen-adapted: independent implementation of object-relative retargeting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .cognitive.store import digest

PHASES = ('approach', 'grasp', 'lift', 'hold')


def transform(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('A finite 4x4 pose is required')
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid homogeneous pose')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or abs(np.linalg.det(rotation) - 1) > 1e-5:
        raise ValueError('Pose rotation must be orthonormal and right-handed')
    return matrix


def relative_pose(world_object_at_phase_start, world_tcp):
    return np.linalg.solve(transform(world_object_at_phase_start), transform(world_tcp))


def retarget_pose(target_object_at_phase_start, object_tcp):
    return transform(target_object_at_phase_start) @ transform(object_tcp)


def _positive_vector(value, name, allow_unknown=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(name + ' requires three components')
    for number in value:
        if not (allow_unknown and number is None):
            if type(number) not in (int, float) or not np.isfinite(number) or number < 0 or (not allow_unknown and number == 0):
                raise ValueError('Invalid ' + name)


def validate(packet):
    if packet.get('schema') != 'explorer-skill-packet-v1' or packet.get('subskill') != 'approach_grasp_lift_hold':
        raise ValueError('Unsupported skill packet')
    if packet.get('units') != {'translation': 'm', 'time': 's', 'jaw_opening': 'm'}:
        raise ValueError('Explicit metric units required')
    if packet.get('frame_convention') != 'T_a_b_maps_b_to_a;right_handed;phase_start_frozen':
        raise ValueError('Unknown frame convention')
    donor = packet['donor']
    if donor.get('robot') != 'so101' or donor.get('behavior_source') not in ('human', 'policy', 'geometric'):
        raise ValueError('Unsupported donor or behavior source')
    if donor.get('domain') not in ('real', 'sim'):
        raise ValueError('Donor domain required')
    if not donor.get('episodes'):
        raise ValueError('Source episode lineage required')
    for source in donor['episodes']:
        if not source.get('id') or len(source.get('sha256', '')) != 64 or any(c not in '0123456789abcdef' for c in source['sha256']):
            raise ValueError('Invalid donor episode fingerprint')
    usage = donor.get('usage', {})
    if not usage.get('reference') or not usage.get('granted_by') or 'adaptation' not in usage.get('allowed', []):
        raise ValueError('Source use permission must be explicit')
    obj = packet['object']
    if not obj.get('instance_id') or not obj.get('class') or not obj.get('surface'):
        raise ValueError('Object instance, class and surface required separately')
    _positive_vector(obj['size_m'], 'object dimensions')
    _positive_vector(obj['size_sigma_m'], 'dimension uncertainty', allow_unknown=True)
    phases = packet.get('phases', [])
    if [p.get('name') for p in phases] != list(PHASES):
        raise ValueError('Only the ordered grasp subskill may be transferred')
    last = -1.
    for phase in phases:
        anchor = transform(phase['world_object_at_start'])
        if phase.get('reference') != 'frozen_at_phase_start' or not phase.get('samples'):
            raise ValueError('Each phase requires a frozen reference and measured samples')
        for sample in phase['samples']:
            stamp = sample.get('time_s')
            if type(stamp) not in (int, float) or not np.isfinite(stamp) or stamp <= last:
                raise ValueError('Packet timestamps must strictly increase')
            last = stamp
            expected = relative_pose(anchor, sample['world_tcp'])
            if not np.allclose(transform(sample['object_tcp']), expected, atol=1e-6):
                raise ValueError('Object-relative pose differs from the recorded measurement')
            jaw = sample.get('jaw_opening_m')
            if type(jaw) not in (int, float) or not np.isfinite(jaw) or not 0 <= jaw <= .2:
                raise ValueError('Metric jaw opening required, not donor motor units')
    quality = packet.get('measurement', {})
    if quality.get('pose_source') not in ('calibrated_measurement', 'simulation_ground_truth'):
        raise ValueError('No measured poses in source')
    if donor['domain'] == 'real' and quality['pose_source'] != 'calibrated_measurement':
        raise ValueError('Simulator truth cannot validate a real donor')
    for key in ('calibration_sha256', 'camera_sha256'):
        if not isinstance(quality.get(key), str) or len(quality[key]) != 64:
            raise ValueError('Measurement version missing: ' + key)
    verification = packet.get('donor_verification', {})
    if verification.get('status') not in ('success', 'failure', 'unverified'):
        raise ValueError('Donor verification status required')
    if verification['status'] == 'success':
        for field, minimum in (('lift_m', .05), ('held_s', 1.)):
            value = verification.get(field)
            if type(value) not in (int, float) or not np.isfinite(value) or value < minimum:
                raise ValueError('Lift and retention evidence required: ' + field)
        methods = ('independent_visual_measurement', 'operator_reviewed_video') if donor['domain'] == 'real' else ('simulation_ground_truth',)
        if verification.get('method') not in methods or not verification.get('evidence_sha256') or not verification.get('evidence_ref'):
            raise ValueError('Independent donor verification required')
    return packet


def admission(packet, *, domain=None):
    validate(packet)
    reasons = []
    if domain is not None and packet['donor']['domain'] != domain:
        reasons.append('donor_domain_mismatch')
    if packet['donor_verification']['status'] != 'success':
        reasons.append('donor_grasp_not_verified')
    measurement = packet['measurement']
    for key in ('position_sigma_m', 'orientation_sigma_rad', 'jaw_sigma_m'):
        value = measurement.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            reasons.append(key + '_unknown')
    if any(v is None for v in packet['object']['size_sigma_m']):
        reasons.append('object_size_uncertainty_unknown')
    return {'eligible': not reasons, 'reasons': reasons, 'packet_sha256': digest(packet),
            'status': 'donor_grasp_verified' if not reasons else 'source_incomplete',
            'recipient_verified': False, 'recipient_policy_trained': False}


def audit_episode(episode):
    """Read-only audit. A whole-task success does not supply missing geometric labels."""
    root = Path(episode['dataset'])
    provenance = json.loads((root / 'provenance.json').read_text())
    hardware = provenance.get('hardware', {})
    rows = [json.loads(line) for line in (root / 'research.jsonl').read_text().splitlines()]
    reasons = []
    if episode.get('outcome') != 'success':
        reasons.append('source_task_not_successful')
    if not rows or any(not r.get('state_valid') for r in rows):
        reasons.append('measured_joint_states_missing')
    if not hardware.get('camera_geometry_valid'):
        reasons.append('camera_geometry_not_verified')
    if not hardware.get('wrist_hand_eye_valid') and not hardware.get('overview_hand_eye_valid'):
        reasons.append('object_to_robot_calibration_missing')
    if not all('world_object_pose' in r and 'world_tcp_pose' in r for r in rows):
        reasons.append('synchronized_object_and_tcp_poses_missing')
    if not all('jaw_opening_m' in r for r in rows):
        reasons.append('metric_jaw_calibration_missing')
    if not all('skill_phase' in r for r in rows):
        reasons.append('grasp_subskill_boundaries_missing')
    if not provenance.get('lift_hold_verification'):
        reasons.append('independent_lift_hold_evidence_missing')
    if not provenance.get('object_instance'):
        reasons.append('object_instance_and_dimensions_missing')
    if not provenance.get('allowed_source_uses'):
        reasons.append('source_use_permission_not_recorded')
    return {'episode_id': episode['id'], 'task': provenance.get('profile', {}).get('task'),
            'outcome': episode['outcome'], 'frames': len(rows), 'eligible_for_packet': not reasons,
            'missing': reasons, 'raw_data_unchanged': True,
            'research_sha256': hashlib.sha256((root / 'research.jsonl').read_bytes()).hexdigest(),
            'provenance_sha256': hashlib.sha256((root / 'provenance.json').read_bytes()).hexdigest(),
            'note': 'A pickup fragment may be used only after its geometry and independent retention evidence are supplied.'}
