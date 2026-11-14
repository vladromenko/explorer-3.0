"""M2: marker pose and cube localisation on synthetic images with known geometry."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from explorer3.simlab.perception import MarkerDetector, TargetTracker, cube_on_table

K = np.array([[415.3, 0, 320], [0, 416.1, 200], [0, 0, 1]])   # Tank camera intrinsics (640x400)


def render_marker(distance, bearing, yaw, side=0.15, size=(400, 640)):
    """Board marker seen from a camera at planar distance/bearing; robot yaw relative to facing."""
    # marker centre in camera frame (x right, z forward); camera height = marker height
    t = np.array([-distance * math.sin(bearing), 0.0, distance * math.cos(bearing)])
    # direction into the board in the camera ground frame: angle (bearing-free) = -yaw
    face = -yaw
    into = np.array([-math.sin(face), 0.0, math.cos(face)])        # (x right, z fwd) of -normal
    normal = -into
    x_m = np.cross([0, -1, 0], normal)                               # marker x to the right on the board
    x_m /= np.linalg.norm(x_m)
    y_m = np.cross(normal, x_m)
    R = np.stack([x_m, y_m, normal], 1)
    h = side / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    pts = (R @ obj.T).T + t
    px = (K @ pts.T).T
    px = px[:, :2] / px[:, 2:3]
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 200)
    canvas = np.full((300, 300), 255, np.uint8)
    canvas[50:250, 50:250] = marker
    src = np.float32([[50, 50], [250, 50], [250, 250], [50, 250]])
    H = cv2.getPerspectiveTransform(src, px.astype(np.float32))
    img = cv2.warpPerspective(canvas, H, (size[1], size[0]), borderValue=180)
    return img


@pytest.mark.parametrize('distance,bearing,yaw', [(0.6, 0.0, 0.0), (1.0, 0.25, -0.3), (1.5, -0.3, 0.4)])
def test_marker_pose_matches_contract_definitions(distance, bearing, yaw):
    det = MarkerDetector(K, side=0.15).detect(render_marker(distance, bearing, yaw))
    assert det is not None
    assert det['distance'] == pytest.approx(distance, abs=0.02 + 0.02 * distance)
    assert det['bearing'] == pytest.approx(bearing, abs=0.02)
    assert det['yaw'] == pytest.approx(yaw, abs=0.12)


def test_tracker_holds_last_detection_and_ages():
    tr = TargetTracker()
    a = tr.update({'distance': 1.0, 'bearing': 0.1, 'yaw': 0.0}, 0.05)
    b = tr.update(None, 0.05)
    assert a['visible'] == 1 and b['visible'] == 0
    assert b['distance'] == 1.0 and b['age_s'] == pytest.approx(0.05)


def test_cube_backprojection_on_table_plane():
    # camera 0.5 m above the table, looking straight down, at base (0.25, 0, 0.5)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    T[:3, 3] = [0.25, 0.0, 0.5]
    cube = np.array([0.30, -0.05, 0.032])
    cam = np.linalg.inv(T) @ np.r_[cube, 1]
    u, v = (K @ cam[:3])[:2] / cam[2]
    img = np.zeros((400, 640, 3), np.uint8)
    cv2.circle(img, (int(round(u)), int(round(v))), 9, (0, 120, 255), -1)   # orange in BGR
    est = cube_on_table(img, K, T, table_z=0.032)
    assert est is not None
    assert np.allclose(est, cube, atol=0.004)
