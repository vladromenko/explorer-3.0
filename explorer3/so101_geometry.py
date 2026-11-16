"""Optional measured-geometry sidecar for the existing SO-101 recording session.

Read-only kinematics and camera evidence. No bus, torque, controller or motion API.
Unvalidated calibration and missing metric jaw measurements remain explicit gaps.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import DATA, ROOT
from .skill_packets import transform

MODEL = ROOT / 'explorer3/so101_vendor/kinematic_data.json'
CALIBRATION = ROOT / 'configs/so101_station/overview_calibration.json'
CONFIG = DATA / 'configs/so101_donor_capture.json'


def configuration():
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {'enabled': False}


def configure_capture(*, enabled, instance_id, marker_side_m, size_m):
    if type(enabled) is not bool or not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError('Укажите отдельный экземпляр предмета')
    size = np.asarray(size_m, dtype=float)
    side = float(marker_side_m)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size < .01) or np.any(size > .1):
        raise ValueError('Нужны три измеренных размера твёрдого предмета от 10 до 100 мм')
    if not np.isfinite(side) or not .005 <= side <= min(size[:2]):
        raise ValueError('Метка должна помещаться на верхней грани; измерьте чёрный квадрат')
    anchor = np.eye(4)
    anchor[2, 3] = -size[2] / 2
    value = {'enabled': enabled, 'instance_id': instance_id.strip(), 'object_class': 'cube',
             'surface': 'rigid_box', 'size_m': size.tolist(), 'marker_id': 7, 'marker_side_m': side,
             'marker_from_object': anchor.tolist(), 'physical_verified': False,
             'scope': 'passive raw evidence only; no verified SkillPacket or movement authorization'}
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(CONFIG)
    return value


def fixed_frames(state, calibration):
    """Measured native joint degrees -> source URDF frames; never command targets."""
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError('Six measured native joints required')
    model = json.loads(MODEL.read_text())
    offsets = np.asarray(calibration['robot_offsets_deg'], dtype=float)
    if offsets.shape != (5,) or not np.isfinite(offsets).all():
        raise ValueError('Five calibrated joint offsets required')
    jaw_zero = calibration['gripper_closed_angle_deg_assumed']
    jaw_span = calibration['gripper_encoder_span_deg']
    if not all(type(x) in (int, float) and np.isfinite(x) for x in (jaw_zero, jaw_span)) or not 0 < jaw_span <= 180:
        raise ValueError('Finite source gripper angular calibration required')
    angles = np.r_[state[:5] + offsets,
                   jaw_zero + jaw_span * state[5] / 100.]
    frames = {'base_link': np.eye(4)}
    for entry, angle in zip(model['joints'], angles):
        origin = np.eye(4)
        origin[:3, :3] = np.asarray(entry['rotation']).reshape(3, 3)
        origin[:3, 3] = entry['translation_m']
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        rotation = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
        frames[entry['child']] = frames[entry['parent']] @ origin @ rotation
    tip = model['tip']
    offset = np.eye(4)
    offset[:3, :3] = np.asarray(tip['rotation']).reshape(3, 3)
    offset[:3, 3] = tip['translation_m']
    frames[tip['name']] = frames[tip['parent']] @ offset
    return frames


def marker_object_pose(rgb, calibration, settings):
    """Independent visual estimate from a known square on a rigid object."""
    import cv2
    if not settings.get('marker_side_m') or 'marker_id' not in settings:
        return {'valid': False, 'reason': 'object_marker_not_configured'}
    side = float(settings['marker_side_m'])
    if not .005 <= side <= .1:
        raise ValueError('Marker size outside supported metric range')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    found = [] if ids is None else [i for i, value in enumerate(ids.flatten()) if int(value) == settings['marker_id']]
    if len(found) != 1:
        return {'valid': False, 'reason': 'object_marker_missing_or_duplicated'}
    points = np.array([[-side/2, side/2, 0], [side/2, side/2, 0], [side/2, -side/2, 0], [-side/2, -side/2, 0.]])
    camera = np.asarray(calibration['camera_matrix'], dtype=float)
    distortion = np.asarray(calibration['distortion'], dtype=float)
    image = corners[found[0]].reshape(4, 2)
    ok, rotations, translations, errors = cv2.solvePnPGeneric(points, image, camera, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return {'valid': False, 'reason': 'marker_pose_unsolved'}
    estimates = []
    for rvec, tvec, error in zip(rotations, translations, np.asarray(errors).flatten()):
        if float(tvec[2, 0]) > 0 and np.isfinite(error):
            matrix = np.eye(4)
            matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
            matrix[:3, 3] = tvec.flatten()
            estimates.append((float(error), matrix))
    estimates.sort(key=lambda item: item[0])
    if not estimates or estimates[0][0] > 3.:
        return {'valid': False, 'reason': 'marker_reprojection_error'}
    gap = estimates[1][0] - estimates[0][0] if len(estimates) > 1 else None
    # Retain ambiguity for review; a small pixel residual is not a calibrated 3D uncertainty.
    pose = np.linalg.inv(transform(calibration['camera_from_robot'])) @ estimates[0][1] @ transform(settings['marker_from_object'])
    return {'valid': True, 'world_object_pose': pose.tolist(), 'marker_corners_px': image.tolist(),
            'reprojection_rmse_px': estimates[0][0], 'alternative_error_gap_px': gap,
            'orientation_ambiguous': gap is None or gap < .2, 'uncertainty_calibrated': False}


class DonorGeometryRecorder:
    def __init__(self, directory, settings):
        self.directory = Path(directory) / 'donor_geometry'
        self.settings = settings
        self.camera = None
        self.lease = None
        self.stream = None
        self.count = 0
        self.calibration = json.loads(CALIBRATION.read_text())
        self.meta = {'kind': 'passive_so101_geometry', 'physical_verified': False,
                     'calibration_sha256': hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
                     'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(), 'settings': settings,
                     'calibration_snapshot': self.calibration, 'calibration_source_json': CALIBRATION.read_text(),
                     'pose_admission': 'unverified; copied camera calibration requires current independent control'}

    def start(self):
        from .hardware import CameraCapture
        from .resources import ResourceLease
        from .so101_vendor.measure import CAMERA
        self.lease = ResourceLease('so101-overview')
        self.lease.__enter__()
        cfg = {'width': 1280, 'height': 720, 'fps': 30, 'camera_formats': {'top': 'MJPG'}}
        self.camera = CameraCapture('top', CAMERA, cfg)
        try:
            self.camera.start()
            self.directory.mkdir(parents=True, exist_ok=False)
            self.stream = (self.directory / 'measurements.jsonl').open('w')
            self.meta['camera'] = dict(self.camera.info)
        except Exception:
            self.close()
            raise
        return self.meta

    def add(self, sample):
        import cv2
        stamp = sample['observation_timestamp_ns']
        frame = self.camera.latest_before(stamp)
        if frame is None or not 0 <= stamp - frame['timestamp_ns'] <= 100_000_000:
            raise ValueError('No fresh synchronized overview evidence')
        frames = fixed_frames(sample['observation.state'], self.calibration)
        ok, image = cv2.imencode('.jpg', cv2.cvtColor(frame['image'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError('Cannot archive overview evidence')
        raw = image.tobytes()
        filename = f'{self.count:06d}.jpg'
        (self.directory / filename).write_bytes(raw)
        row = {'frame_index': self.count, 'observation_timestamp_ns': stamp, 'camera_timestamp_ns': frame['timestamp_ns'],
               'camera_sequence': frame['sequence'], 'measured_native_joints': sample['observation.state'],
               'world_fixed_tip_pose': frames['gripper_frame_link'].tolist(),
               'tip_definition': 'source URDF fixed-jaw tip, not yet a calibrated contact-midpoint TCP',
               'jaw_opening_m': None, 'metric_jaw_calibrated': False, 'image': filename,
               'image_sha256': hashlib.sha256(raw).hexdigest(),
               'visual': marker_object_pose(frame['image'], self.calibration, self.settings)}
        self.stream.write(json.dumps(row) + '\n')
        self.count += 1

    def close(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.camera:
            self.camera.close()
            self.camera = None
        if self.lease:
            self.lease.__exit__(None, None, None)
            self.lease = None
        self.meta['frames'] = self.count
        if self.directory.exists():
            (self.directory / 'provenance.json').write_text(json.dumps(self.meta, indent=2))
        return self.meta
