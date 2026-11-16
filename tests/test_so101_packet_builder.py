"""Synthetic geometry fixtures exercise artifact guards, never physical acceptance."""
import copy
import hashlib
import json

import numpy as np
import pytest

from explorer3 import catalog, datasets, knowledge, so101_geometry as geometry, so101_packet_builder as builder


@pytest.fixture
def recording(tmp_path, monkeypatch):
    for module in (catalog, datasets, knowledge, builder):
        monkeypatch.setattr(module, 'DATA', tmp_path)
    monkeypatch.setattr(catalog, 'prepare', lambda: None)
    monkeypatch.setattr(knowledge, 'STORE', tmp_path / 'world/knowledge.json')
    catalog.initialize()
    root = tmp_path / 'datasets/so101_pick_place/fixture'
    directory = root / 'donor_geometry'
    directory.mkdir(parents=True)
    raw = geometry.CALIBRATION.read_text()
    source = json.loads(raw)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    meta = {'frames': 100, 'calibration_sha256': sha, 'calibration_source_json': raw,
            'model_sha256': hashlib.sha256(geometry.MODEL.read_bytes()).hexdigest(), 'camera': {'device': 'fixture'},
            'settings': {'instance_id': 'fixture-only', 'object_class': 'cube', 'surface': 'rigid_box', 'size_m': [.032]*3}}
    (directory / 'provenance.json').write_text(json.dumps(meta))
    (root / 'provenance.json').write_text(json.dumps({'domain': 'real', 'fixture': True}))
    rows, original = [], []
    for i in range(100):
        state = [0, 30 if i < 40 else max(0., 30*(60-i)/20), 0, 0, 0, 50]
        pose = geometry.fixed_frames(state, source)['gripper_frame_link']
        obj = np.eye(4)
        obj[:3, 3] = pose[:3, 3] - [0, 0, .03]
        stamp = 1_000_000_000 + i*50_000_000
        image = f'{i:06d}.jpg'
        (directory / image).write_bytes(b'isolated fixture image')
        rows.append({'frame_index': i, 'observation_timestamp_ns': stamp, 'camera_timestamp_ns': stamp,
                     'measured_native_joints': state, 'world_fixed_tip_pose': pose.tolist(),
                     'image': image, 'image_sha256': hashlib.sha256(b'isolated fixture image').hexdigest(),
                     'visual': {'valid': True, 'orientation_ambiguous': False, 'world_object_pose': obj.tolist()}})
        original.append({'observation_timestamp_ns': stamp, 'observation.state': state, 'command_source': 'human', 'state_valid': True})
    (directory / 'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root / 'research.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in original))
    catalog.add_episode({'id': 'fixture', 'profile': 'so101_pick_place', 'dataset': str(root), 'episode_index': 0,
                         'session': 'fixture', 'scene': 'fixture', 'category': 'demonstration', 'outcome': 'success', 'frames': 100})
    controls = {'kind': 'independent_geometry_controls', 'used_for_fit': False, 'source_calibration_sha256': sha,
                'pose_controls': [{'id': str(i), 'predicted_m': [i*.1, 0, 0], 'measured_m': [i*.1+.001, 0, 0], 'measurement_sigma_m': .001} for i in range(3)],
                'jaw_controls': [{'id': str(i), 'predicted_m': i*.03, 'measured_m': i*.03+.001, 'measurement_sigma_m': .001} for i in range(3)]}
    control_file = tmp_path / 'controls.json'
    control_file.write_text(json.dumps(controls))
    calibration = {'schema': 'so101-donor-geometry-validation-v1', 'source_calibration_sha256': sha,
                   'source_model_sha256': meta['model_sha256'], 'camera_device': 'fixture', 'camera_sha256': builder.camera_fingerprint(meta['camera']), 'operator': 'fixture-only',
                   'independent_control_evidence': {'reference': 'controls.json', 'sha256': hashlib.sha256(control_file.read_bytes()).hexdigest()},
                   'fixed_tip_from_tcp': np.eye(4).tolist(), 'jaw_curve_percent_m': [[0, 0], [50, .06], [100, .12]],
                   'position_sigma_m': .002, 'orientation_sigma_rad': .05, 'jaw_sigma_m': .002,
                   'size_sigma_m': [.001]*3, 'closing_axis_tcp': [-1, 0, 0], 'camera_delay_s': 0., 'camera_delay_sigma_s': .001}
    calibration_file = tmp_path / 'calibration.json'
    calibration_file.write_text(json.dumps(calibration))
    review = {'episode_sha256': datasets.fingerprint(root), 'operator': 'fixture-only',
              'calibration_ref': 'calibration.json', 'calibration_sha256': hashlib.sha256(calibration_file.read_bytes()).hexdigest(),
              'phase_frames': dict(zip(('approach', 'grasp', 'lift', 'hold'), [[0,20], [20,40], [40,60], [60,100]])),
              'source_use': {'granted_by': 'fixture-only', 'reference': 'unit-test', 'allowed': ['adaptation', 'training']}}
    return root, meta, calibration, review


def test_reviewed_fixture_builds_packet_without_altering_original(recording):
    root, _, _, review = recording
    before = datasets.fingerprint(root)
    item, packet = builder.build_real_packet('fixture', review)
    assert item['verification_status'] == 'donor_grasp_verified'
    assert not item['recipient_verified'] and not item['recipient_policy_trained']
    assert packet['donor_verification']['held_s'] >= 1
    assert packet['measurement']['synchronization'].startswith('bracketed measured joints')
    assert datasets.fingerprint(root) == before
    assert knowledge.load_packet(item['skill_packet_sha256']) == packet
    (root / 'donor_geometry/000020.jpg').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='camera evidence changed'):
        builder.build_real_packet('fixture', review)


def test_unknown_geometry_and_fit_reuse_cannot_validate_source(recording):
    root, meta, calibration, review = recording
    path = root.parents[2] / 'calibration.json'
    calibration.pop('camera_delay_s')
    path.write_text(json.dumps(calibration))
    changed = {**review, 'calibration_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='camera read-to-exposure'):
        builder.build_real_packet('fixture', changed)
    controls = root.parents[2] / 'controls.json'
    value = json.loads(controls.read_text())
    value['used_for_fit'] = True
    controls.write_text(json.dumps(value))
    calibration['camera_delay_s'] = 0.
    calibration['independent_control_evidence']['sha256'] = hashlib.sha256(controls.read_bytes()).hexdigest()
    path.write_text(json.dumps(calibration))
    changed['calibration_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='excluded from its fit'):
        builder.build_real_packet('fixture', changed)


def test_frame_bracketing_and_source_phase_reference_are_required(recording):
    root, _, _, review = recording
    path = root / 'donor_geometry/measurements.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['camera_timestamp_ns'] -= 10_000_000
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    changed = {**review, 'episode_sha256': datasets.fingerprint(root)}
    with pytest.raises(ValueError, match='bracketing'):
        builder.build_real_packet('fixture', changed)
    rows[0]['camera_timestamp_ns'] += 10_000_000
    rows[20]['visual']['orientation_ambiguous'] = True
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    changed['episode_sha256'] = datasets.fingerprint(root)
    with pytest.raises(ValueError, match='unambiguous'):
        builder.build_real_packet('fixture', changed)
