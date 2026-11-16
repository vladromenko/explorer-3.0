"""Turn reviewed real donor measurements into an immutable SkillPacket.

Raw recordings are preserved. A successful episode label alone cannot pass this
builder: current geometric validation, metric jaw samples and visual lift are needed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import catalog, datasets, knowledge
from .paths import DATA
from .skill_packets import PHASES, transform, relative_pose, admission
from .cognitive.store import digest


def read_artifact(reference, sha):
    path = (DATA / reference).resolve()
    if not path.is_relative_to(DATA) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError('Measured evidence is missing or changed')
    return json.loads(path.read_text())


def camera_fingerprint(camera):
    keys = ('device', 'device_resolved', 'width', 'height', 'fps_reported',
            'pixel_format_requested', 'pixel_format_reported', 'controls_readback_before', 'controls_readback_after')
    return digest({key: camera.get(key) for key in keys})


def calibration_for_review(review, meta):
    calibration = read_artifact(review['calibration_ref'], review['calibration_sha256'])
    if calibration.get('schema') != 'so101-donor-geometry-validation-v1':
        raise ValueError('Independent current geometry validation required')
    if calibration.get('source_calibration_sha256') != meta['calibration_sha256'] or calibration.get('source_model_sha256') != meta['model_sha256']:
        raise ValueError('Validation belongs to another camera/model calibration')
    if calibration.get('camera_device') != meta['camera']['device'] or calibration.get('camera_sha256') != camera_fingerprint(meta['camera']):
        raise ValueError('Validation belongs to another camera')
    if not calibration.get('operator') or not calibration.get('independent_control_evidence'):
        raise ValueError('Independent physical control evidence required')
    evidence = calibration['independent_control_evidence']
    controls = read_artifact(evidence['reference'], evidence['sha256'])
    if controls.get('kind') != 'independent_geometry_controls' or controls.get('used_for_fit') is not False or controls.get('source_calibration_sha256') != meta['calibration_sha256']:
        raise ValueError('Held-out controls must match this calibration and be excluded from its fit')
    # Engineering admission for this pilot only, not a general accuracy guarantee.
    for key, dimension, limit in (('pose_controls', 3, .005), ('jaw_controls', 1, .003)):
        items = controls.get(key, [])
        if len(items) < 3 or len({x.get('id') for x in items}) != len(items) or any(not x.get('id') for x in items):
            raise ValueError('At least three distinct measured controls required: ' + key)
        for item in items:
            predicted = np.asarray(item['predicted_m'], dtype=float).reshape(-1)
            measured = np.asarray(item['measured_m'], dtype=float).reshape(-1)
            sigma = item.get('measurement_sigma_m')
            if predicted.shape != (dimension,) or measured.shape != (dimension,) or not np.isfinite(predicted).all() or not np.isfinite(measured).all():
                raise ValueError('Finite independent measured controls required')
            if type(sigma) not in (int, float) or not np.isfinite(sigma) or not 0 < sigma <= limit:
                raise ValueError('Independent reference measurement uncertainty required')
            if np.linalg.norm(predicted - measured) + sigma > limit:
                raise ValueError('Independent control error exceeds pilot tolerance: ' + key)
    delay, delay_sigma = calibration.get('camera_delay_s'), calibration.get('camera_delay_sigma_s')
    if type(delay) not in (int, float) or not np.isfinite(delay) or not 0 <= delay <= .25:
        raise ValueError('Measured camera read-to-exposure delay required')
    if type(delay_sigma) not in (int, float) or not np.isfinite(delay_sigma) or not 0 < delay_sigma <= .02:
        raise ValueError('Measured camera delay uncertainty required')
    transform(calibration['fixed_tip_from_tcp'])
    curve = np.asarray(calibration['jaw_curve_percent_m'], dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 3 or not np.isfinite(curve).all():
        raise ValueError('Measured metric gripper curve required')
    if np.any(np.diff(curve, axis=0) <= 0) or curve[0, 0] < 0 or curve[-1, 0] > 100 or curve[0, 1] < 0 or curve[-1, 1] > .2:
        raise ValueError('Invalid metric gripper curve')
    for key, maximum in (('position_sigma_m', .005), ('orientation_sigma_rad', .2), ('jaw_sigma_m', .003)):
        value = calibration.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= maximum:
            raise ValueError('Measured uncertainty outside pilot scope: ' + key)
    return calibration


def build_real_packet(episode_id, review):
    entries = [e for e in catalog.rows('episodes') if e['id'] == episode_id]
    if len(entries) != 1 or entries[0]['profile'] != 'so101_pick_place' or entries[0]['outcome'] != 'success' or entries[0]['category'] not in ('demonstration', 'correction') or entries[0]['excluded']:
        raise ValueError('A successful original real SO-101 episode is required')
    entry = entries[0]
    root = Path(entry['dataset']).resolve()
    fingerprint = datasets.fingerprint(root)
    if review.get('episode_sha256') != fingerprint or not review.get('operator'):
        raise ValueError('Review must identify the unchanged original episode and reviewer')
    provenance = json.loads((root / 'provenance.json').read_text())
    if provenance.get('domain', 'real') != 'real':
        raise ValueError('A simulated donor cannot become a real packet')
    directory = root / 'donor_geometry'
    meta = json.loads((directory / 'provenance.json').read_text())
    from .so101_geometry import MODEL, fixed_frames
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != meta['model_sha256']:
        raise ValueError('Source kinematic model version unavailable')
    raw_calibration = meta.get('calibration_source_json', '')
    if hashlib.sha256(raw_calibration.encode()).hexdigest() != meta['calibration_sha256']:
        raise ValueError('Source calibration snapshot changed')
    source_calibration = json.loads(raw_calibration)
    calibration = calibration_for_review(review, meta)
    rows = [json.loads(line) for line in (directory / 'measurements.jsonl').read_text().splitlines()]
    original = [json.loads(line) for line in (root / 'research.jsonl').read_text().splitlines()]
    if not rows or len(rows) != len(original) or len(rows) != meta['frames']:
        raise ValueError('Geometry and original recording do not align')
    for index, (row, sample) in enumerate(zip(rows, original)):
        if row['frame_index'] != index or row['observation_timestamp_ns'] != sample['observation_timestamp_ns'] or row['measured_native_joints'] != sample['observation.state']:
            raise ValueError('Geometry does not match the recorded measured state')
        if sample.get('command_source') != 'human' or sample.get('state_valid') is not True:
            raise ValueError('Measured human demonstration required')
        expected_tip = fixed_frames(row['measured_native_joints'], source_calibration)['gripper_frame_link']
        if not np.allclose(transform(row['world_fixed_tip_pose']), expected_tip, atol=1e-8):
            raise ValueError('Fixed tip does not match the measured kinematics')
        image = (directory / row['image']).resolve()
        if not image.is_relative_to(directory) or hashlib.sha256(image.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Independent camera evidence changed')
        age = row['observation_timestamp_ns'] - row['camera_timestamp_ns']
        if not 0 <= age <= 100_000_000 or (index and row['observation_timestamp_ns'] <= rows[index-1]['observation_timestamp_ns']):
            raise ValueError('Invalid source measurement timing')
    stamps = np.asarray([r['observation_timestamp_ns'] for r in rows], dtype=np.int64)
    def aligned_joints(row):
        stamp = row['camera_timestamp_ns'] - round(calibration['camera_delay_s'] * 1e9)
        index = int(np.searchsorted(stamps, stamp, side='left'))
        if index < len(stamps) and stamps[index] == stamp:
            return np.asarray(rows[index]['measured_native_joints'], dtype=float)
        if index == 0 or index >= len(stamps) or stamps[index] - stamps[index-1] > 100_000_000:
            raise ValueError('Camera exposure has no close bracketing joint measurements')
        fraction = (stamp - stamps[index-1]) / (stamps[index] - stamps[index-1])
        before = np.asarray(rows[index-1]['measured_native_joints'], dtype=float)
        after = np.asarray(rows[index]['measured_native_joints'], dtype=float)
        return before + fraction * (after - before)
    settings = meta['settings']
    phase_ranges = review.get('phase_frames', {})
    if set(phase_ranges) != set(PHASES):
        raise ValueError('Review all four grasp phases')
    curve = np.asarray(calibration['jaw_curve_percent_m'])
    offset = transform(calibration['fixed_tip_from_tcp'])
    phases, trace = [], []
    previous_end, held_s, best_held, best_lift, previous_time = 0, 0., 0., 0., None
    previous_retained = False
    initial_height = None
    last_selected_stamp = -1
    for name in PHASES:
        start, end = phase_ranges[name]
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(rows):
            raise ValueError('Phase boundaries must be ordered recorded frame indices')
        previous_end = end
        selected = []
        for row in rows[start:end]:
            if row['camera_timestamp_ns'] > last_selected_stamp:
                selected.append(row)
                last_selected_stamp = row['camera_timestamp_ns']
        if len(selected) < 2:
            raise ValueError('Each phase needs at least two distinct overview frames')
        first = selected[0]['visual']
        if not first.get('valid') or first.get('orientation_ambiguous') is not False:
            raise ValueError('Phase reference needs a visible unambiguous measured object')
        anchor = transform(first['world_object_pose'])
        initial_height = float(anchor[2, 3]) if initial_height is None else initial_height
        phase = {'name': name, 'reference': 'frozen_at_phase_start', 'world_object_at_start': anchor.tolist(), 'samples': []}
        for row in selected:
            t = row['camera_timestamp_ns'] / 1e9 - calibration['camera_delay_s']
            joints = aligned_joints(row)
            tcp = fixed_frames(joints, source_calibration)['gripper_frame_link'] @ offset
            grip = joints[5]
            if not curve[0, 0] <= grip <= curve[-1, 0]:
                raise ValueError('Recorded gripper position outside measured metric curve; no extrapolation')
            jaw = float(np.interp(grip, curve[:, 0], curve[:, 1]))
            phase['samples'].append({'time_s': t, 'world_tcp': tcp.tolist(),
                                    'object_tcp': relative_pose(anchor, tcp).tolist(), 'jaw_opening_m': jaw})
            visual = row['visual']
            valid = visual.get('valid') and not visual.get('orientation_ambiguous', True)
            lift, retained = None, False
            if valid:
                position = transform(visual['world_object_pose'])[:3, 3]
                lift = float(position[2] - initial_height)
                retained = lift >= .05 and np.linalg.norm(position - tcp[:3, 3]) < .06
            camera_time = row['camera_timestamp_ns'] / 1e9
            dt = camera_time - previous_time if previous_time is not None else 0.
            if dt > 0 or previous_time is None:
                held_s = held_s + dt if retained and previous_retained and dt <= .15 else 0.
                previous_retained = bool(retained)
                previous_time = camera_time
            elif dt < 0:
                raise ValueError('Overview camera timestamp moved backwards')
            if held_s > best_held:
                best_held, best_lift = held_s, lift
            trace.append({'time_s': t, 'lift_m': lift, 'held_s': held_s, 'visual_valid': bool(valid)})
        phases.append(phase)
    samples = [sample for phase in phases for sample in phase['samples']]
    speeds = [np.linalg.norm(np.asarray(b['world_tcp'])[:3, 3] - np.asarray(a['world_tcp'])[:3, 3]) / (b['time_s'] - a['time_s'])
              for a, b in zip(samples, samples[1:])]
    timing_sigma = max(speeds, default=0.) * calibration['camera_delay_sigma_s']
    position_sigma = float(np.hypot(calibration['position_sigma_m'], timing_sigma))
    if position_sigma > .005:
        raise ValueError('Measured motion and camera timing uncertainty exceed the 5 mm pilot scope')
    if best_held < 1.:
        raise ValueError('No independently visible 50 mm lift held for one continuous second')
    verification = {'method': 'independent_visual_measurement', 'episode_id': episode_id,
                    'episode_sha256': fingerprint, 'review': review, 'trace': trace,
                    'lift_m': best_lift, 'held_s': best_held, 'recipient_verified': False}
    path = DATA / 'world/skill_verifications' / (digest(verification) + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verification, indent=2, allow_nan=False))
    source_tcp = transform(phases[1]['samples'][0]['world_tcp'])
    source_object = transform(phases[1]['world_object_at_start'])
    axis = source_object[:3, :3].T @ source_tcp[:3, :3] @ np.asarray(calibration['closing_axis_tcp'])
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < .99:
        raise ValueError('Measured closing axis required')
    axis /= np.linalg.norm(axis)
    packet = {'schema': 'explorer-skill-packet-v1', 'subskill': 'approach_grasp_lift_hold',
        'units': {'translation': 'm', 'time': 's', 'jaw_opening': 'm'},
        'frame_convention': 'T_a_b_maps_b_to_a;right_handed;phase_start_frozen',
        'donor': {'robot': 'so101', 'behavior_source': 'human', 'domain': 'real',
                  'episodes': [{'id': episode_id, 'sha256': fingerprint}], 'usage': review['source_use']},
        'object': {'instance_id': settings['instance_id'], 'class': settings['object_class'], 'surface': settings['surface'],
                   'size_m': settings['size_m'], 'size_sigma_m': calibration['size_sigma_m']},
        'phases': phases,
        'measurement': {'pose_source': 'calibrated_measurement', 'calibration_sha256': review['calibration_sha256'],
                        'camera_sha256': camera_fingerprint(meta['camera']), 'closing_axis_object': axis.tolist(),
                        'tcp_definition': 'reviewed calibrated fixed contact reference; not uncorrected URDF tip',
                        'synchronization': 'bracketed measured joints interpolated to distinct overview timestamps minus independently measured camera delay',
                        'camera_delay_s': calibration['camera_delay_s'], 'camera_delay_sigma_s': calibration['camera_delay_sigma_s'],
                        **{key: calibration[key] for key in ('orientation_sigma_rad', 'jaw_sigma_m')},
                        'position_sigma_m': position_sigma, 'timing_position_sigma_m': timing_sigma},
        'donor_verification': {'status': 'success', 'lift_m': best_lift, 'held_s': best_held,
                              'method': 'independent_visual_measurement', 'evidence_ref': str(path.relative_to(DATA)),
                              'evidence_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    admission(packet, domain='real')
    return knowledge.add_packet(packet), packet
