import hashlib
import json

import cv2
import numpy as np
import pytest

from explorer3 import so101_geometry as geometry


def test_known_marker_geometry_and_missing_detection_are_distinct():
    calibration = json.loads(geometry.CALIBRATION.read_text())
    marker_from_object = np.eye(4)
    marker_from_object[2, 3] = -.016
    settings = {'marker_id': 7, 'marker_side_m': .02, 'marker_from_object': marker_from_object.tolist()}
    world_marker = np.eye(4)
    world_marker[:3, 3] = [.22, .02, .032]
    camera_marker = np.asarray(calibration['camera_from_robot']) @ world_marker
    square = np.array([[-.01, .01, 0], [.01, .01, 0], [.01, -.01, 0], [-.01, -.01, 0]])
    corners, _ = cv2.projectPoints(square, cv2.Rodrigues(camera_marker[:3, :3])[0], camera_marker[:3, 3],
                                   np.asarray(calibration['camera_matrix']), np.asarray(calibration['distortion']))
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 600)
    warp = cv2.getPerspectiveTransform(np.float32([[0, 0], [599, 0], [599, 599], [0, 599]]), corners.reshape(4, 2).astype(np.float32))
    grey = cv2.warpPerspective(marker, warp, (1280, 720), borderValue=255)
    result = geometry.marker_object_pose(np.repeat(grey[:, :, None], 3, axis=2), calibration, settings)
    assert result['valid'] and not result['uncertainty_calibrated']
    np.testing.assert_allclose(np.asarray(result['world_object_pose'])[:3, 3], [.22, .02, .016], atol=.015)
    missing = geometry.marker_object_pose(np.full((720, 1280, 3), 255, np.uint8), calibration, settings)
    assert missing == {'valid': False, 'reason': 'object_marker_missing_or_duplicated'}


def test_fixed_tip_uses_measured_angles_offsets_and_distinct_jaw_frame():
    calibration = json.loads(geometry.CALIBRATION.read_text())
    a = geometry.fixed_frames([0, -30, 40, 10, 0, 0], calibration)
    b = geometry.fixed_frames([0, -30, 40, 10, 0, 50], calibration)
    np.testing.assert_allclose(a['gripper_frame_link'], b['gripper_frame_link'])
    assert not np.allclose(a['moving_jaw_so101_v1_link'], b['moving_jaw_so101_v1_link'])
    c = geometry.fixed_frames([0, -30, 40, 10, 0, 0], {**calibration, 'robot_offsets_deg': [0]*5})
    assert np.linalg.norm(c['gripper_frame_link'][:3, 3] - a['gripper_frame_link'][:3, 3]) > .005
    with pytest.raises(ValueError, match='measured native'):
        geometry.fixed_frames([float('nan')]*6, calibration)


def test_raw_recorder_keeps_causality_images_and_unknown_calibration(tmp_path, monkeypatch):
    from explorer3 import hardware, resources
    monkeypatch.setattr(resources, 'DATA', tmp_path)
    class Camera:
        info = {'device': 'fixture', 'camera_geometry_valid': False}
        frame = {'timestamp_ns': 1_000_000_000, 'sequence': 1, 'image': np.zeros((720, 1280, 3), np.uint8)}
        def __init__(self, *a):
            self.closed = False
        def start(self):
            pass
        def latest_before(self, cutoff):
            return self.frame
        def close(self):
            self.closed = True
    monkeypatch.setattr(hardware, 'CameraCapture', Camera)
    recorder = geometry.DonorGeometryRecorder(tmp_path / 'episode', {'enabled': True})
    recorder.start()
    sample = {'observation_timestamp_ns': 1_050_000_000, 'observation.state': [0]*6}
    recorder.add(sample)
    sample['observation_timestamp_ns'] = 1_200_000_000
    with pytest.raises(ValueError, match='fresh synchronized'):
        recorder.add(sample)
    meta = recorder.close()
    row = json.loads((recorder.directory / 'measurements.jsonl').read_text())
    assert row['jaw_opening_m'] is None and not row['metric_jaw_calibrated']
    assert not row['visual']['valid'] and not meta['physical_verified']
    assert row['image_sha256'] == hashlib.sha256((recorder.directory / row['image']).read_bytes()).hexdigest()
    assert meta['frames'] == 1


def test_configuration_is_only_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry, 'CONFIG', tmp_path / 'capture.json')
    cfg = geometry.configure_capture(enabled=True, instance_id='cube-1', marker_side_m=.02, size_m=[.032]*3)
    assert cfg['enabled'] and not cfg['physical_verified']
    assert geometry.configuration() == cfg
    with pytest.raises(ValueError, match='метка|Метка'):
        geometry.configure_capture(enabled=True, instance_id='cube-1', marker_side_m=.05, size_m=[.032]*3)


def test_stale_optional_geometry_stops_existing_session_before_finalize(tmp_path, monkeypatch):
    from explorer3 import catalog, resources, session as sessions
    from explorer3.control import Control
    monkeypatch.setattr(resources, 'DATA', tmp_path)
    monkeypatch.setattr(catalog, 'DATA', tmp_path)
    catalog.initialize()
    monkeypatch.setattr(geometry, 'configuration', lambda: {'enabled': True})
    events = []
    class Source:
        def start(self, **kw):
            events.append('source_start')
            return {}
        def begin_episode(self):
            events.append('begin')
        def sample(self, *a, **kw):
            events.append('sample')
            return {'teleoperation_phase': 'following'}
        def close(self):
            events.append('source_closed')
            return []
    class Writer:
        id = 'fixture'
        root = tmp_path
        def __init__(self, *a, **kw):
            pass
        def add(self, sample):
            raise AssertionError('Stale geometry cannot become a usable record')
        def finish(self, outcome, provenance):
            events.append(outcome)
            assert provenance['donor_geometry']['physical_verified'] is False
    class Geometry:
        def __init__(self, *a):
            pass
        def start(self):
            events.append('geometry_start')
            return {'physical_verified': False}
        def add(self, sample):
            raise ValueError('stale geometry')
        def close(self):
            events.append('geometry_closed')
            return {'physical_verified': False}
    monkeypatch.setattr(sessions, 'HardwareSource', lambda **kw: Source())
    monkeypatch.setattr(sessions, 'EpisodeWriter', Writer)
    monkeypatch.setattr(geometry, 'DonorGeometryRecorder', Geometry)
    control = Control()
    control.stop_event.clear()
    control.expires = float('inf')
    session = sessions.Session(control)
    session.record_requested = True
    session._run('so101_pick_place', 'fixture', 'fixture')
    assert events == ['source_start', 'geometry_start', 'begin', 'sample', 'source_closed', 'geometry_closed', 'aborted']
    assert control.stop_event.is_set() and 'stale geometry' in session.error
