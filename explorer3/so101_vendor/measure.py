"""Measure an object on the SO-101 table from the fixed overview camera (no motion).

SAM 3 (Explorer's own copy, perception.analyze) gives the object's silhouette; the copied
overview hand-eye calibration and table projection map it to metres on the table plane.
An upright box (all eight vertices projected, as in the source project) is fitted to the silhouette,
so side faces are not mistaken for width; when the source project's cube fit explains the same
silhouette, it is used. Check 2026-11-16: blue cube 30.1 mm (source project measured 29.5 mm).
"""
import json
import time
from pathlib import Path

import numpy as np

from ..paths import DATA
from .table_projection import TableProjection

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / 'configs' / 'so101_station' / 'overview_calibration.json'
CAMERA = '/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_D4EA612F-video-index0'


def projection(z=0.):
    c = json.loads(CALIBRATION.read_text())
    if not c.get('validated'):
        raise RuntimeError('калибровка верхней камеры SO-101 не подтверждена')
    return TableProjection(np.array(c['camera_matrix']), np.array(c['distortion']), np.array(c['camera_from_robot']),
                           table_z_m=c['table_z_m'] + z, reprojection_rmse_px=c['reprojection_rmse_px'])


def capture(path):
    import cv2
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError('верхняя камера SO-101 занята или не подключена')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0); cap.set(cv2.CAP_PROP_FOCUS, 0)
        frame = None
        for _ in range(12):           # let exposure settle
            ok, frame = cap.read()
        if frame is None or frame.shape[:2] != (720, 1280):
            raise RuntimeError('верхняя камера SO-101 не дала кадр 1280×720')
        cv2.imwrite(str(path), frame)
    finally:
        cap.release()


def footprint(mask, z):
    """Minimum-area rectangle (metres) of the mask's outline projected on the plane at height z."""
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)[:, 0, :]
    outline = outline[np.linspace(0, len(outline) - 1, min(len(outline), 200)).astype(int)]
    proj = projection(z)
    pts = np.array([proj.pixel_to_table((float(u), float(v))).xyz_m[:2] for u, v in outline], dtype=np.float32)
    (cx, cy), (a, b), yaw = cv2.minAreaRect(pts)
    return {'center_m': [float(cx), float(cy)], 'sides_m': sorted([float(a), float(b)]), 'yaw_deg': float(yaw)}


def measure(prompt, image=None, device='cuda'):
    """Returns {'ok', 'width_m', 'length_m', 'height_m', 'center_m', 'score', 'image', ...} or {'ok': False, 'reason'}."""
    from ..perception import analyze
    out = DATA / 'so101_station' / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    if image is None:
        image = out / 'overview.jpg'
        capture(image)
    rep = analyze(image, out, 'sam3', prompt, device=device)
    if not rep['scores']:
        return {'ok': False, 'reason': f'SAM 3 не нашёл «{prompt}» на столе SO-101', 'dir': str(out)}
    masks = np.load(rep['masks'])['masks']
    i = int(np.argmax(rep['scores']))
    if masks[i].sum() < 150:
        return {'ok': False, 'reason': 'силуэт предмета слишком мал для измерения', 'dir': str(out)}
    from .cuboid_vision import fit_upright_cube, fit_upright_cuboid
    calib = json.loads(CALIBRATION.read_text())
    try:
        box = fit_upright_cuboid(masks[i], calib)
    except ValueError as exc:
        return {'ok': False, 'reason': f'силуэт не годится для измерения: {exc}', 'dir': str(out)}
    model = 'cuboid'
    try:                                  # the source project's validated cube fit, when it explains the silhouette
        _, ev = fit_upright_cube([(masks[i], calib)])
        if abs(ev['size_m'] - box['width_m']) < .004:
            box.update(width_m=ev['size_m'], length_m=ev['size_m'], height_m=ev['size_m'], height_measured=True,
                       silhouette_rmse_px=ev['silhouette_rmse_px'])
            model = 'cube'
    except ValueError:
        pass
    w = box['width_m']
    if not (.005 < w < .3):
        return {'ok': False, 'reason': f'нереальный размер {w:.3f} м', 'dir': str(out)}
    result = {'ok': True, 'model': model, 'width_m': round(w, 4), 'length_m': round(box['length_m'], 4),
              'height_m': round(box['height_m'], 4), 'height_sigma_m': box.get('height_sigma_m'),
              'center_m': box['center_m'], 'yaw_rad': box['yaw_rad'], 'silhouette_rmse_px': round(box['silhouette_rmse_px'], 2),
              'score': float(rep['scores'][i]), 'image': str(image), 'overlay': rep['overlay'], 'dir': str(out),
              'calibration': str(CALIBRATION)}
    (out / 'measure.json').write_text(json.dumps(result, indent=1))
    return result
