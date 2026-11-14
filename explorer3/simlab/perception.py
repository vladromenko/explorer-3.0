"""M2: perception shared by simulation checks and the real robots.

Bases: ArUco target marker -> contract target (planar distance, bearing, yaw) in the
camera-centred ground frame, exactly as explorer3.simlab.dock_gpu defines them:
  distance = planar distance camera -> marker centre,
  bearing  = angle of the marker centre left of the camera axis (CCW positive),
  yaw      = robot heading minus the direction facing into the board (0 when square on).
Arms: coloured cube on a known table plane seen by a calibrated camera -> cube position
in the arm base frame.
OpenCV camera frame: x right, y down, z forward.
"""
import math

import cv2
import numpy as np

DICTS = {'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_5X5_50': cv2.aruco.DICT_5X5_50}


class MarkerDetector:
    def __init__(self, K, dist=None, side=0.10, dictionary='DICT_4X4_50', marker_id=0):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
        self.side, self.marker_id = side, marker_id
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTS[dictionary]), params)
        h = side / 2
        self.object = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

    def detect(self, image):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel()):
            if int(i) != self.marker_id:
                continue
            n, rvecs, tvecs, errors = cv2.solvePnPGeneric(self.object, c.reshape(4, 2).astype(np.float64), self.K,
                                                          self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not n:
                return None
            # Planar-pose ambiguity (small or distant markers): the board is vertical and the
            # camera roughly level, so keep the solution whose normal is most horizontal and
            # points away from the camera; reprojection error breaks remaining ties.
            best = None
            for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errors)):
                R, _ = cv2.Rodrigues(rvec)
                into = -R[:, 2]
                score = abs(into[1]) + (0.0 if into @ tvec.ravel() > 0 else 10.0) + 0.01 * float(err)
                if best is None or score < best[0]:
                    best = (score, R, tvec.ravel())
            _, R, t = best
            into_board = -R[:, 2]                         # marker normal points out of the board
            return {'distance': float(math.hypot(t[0], t[2])), 'bearing': float(math.atan2(-t[0], t[2])),
                    'yaw': float(-math.atan2(-into_board[0], into_board[2])), 'visible': 1, 'age_s': 0.0,
                    'corners_px': c.reshape(4, 2).tolist()}
        return None


class TargetTracker:
    """Holds the last valid detection and its age, like the simulator (no dead reckoning)."""

    def __init__(self, age_clip=2.0):
        self.last, self.age, self.clip = {'distance': 0.0, 'bearing': 0.0, 'yaw': 0.0}, age_clip, age_clip

    def update(self, detection, dt):
        if detection is not None:
            self.last = {k: detection[k] for k in ('distance', 'bearing', 'yaw')}
            self.age = 0.0
            visible = 1
        else:
            self.age = min(self.clip, self.age + dt)
            visible = 0
        return {**self.last, 'visible': visible, 'age_s': self.age}


def cube_on_table(image_bgr, K, cam_to_base, table_z, hsv_low=(5, 120, 120), hsv_high=(22, 255, 255), min_area=40):
    """Centroid of the coloured cube top face, back-projected onto the plane z = table_z
    (cube top height in the base frame). cam_to_base: 4x4 camera pose in the arm base frame."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return None
    u, v = cents[k]
    ray_cam = np.linalg.solve(np.asarray(K, float), np.array([u, v, 1.0]))
    T = np.asarray(cam_to_base, float)
    origin, direction = T[:3, 3], T[:3, :3] @ ray_cam
    if abs(direction[2]) < 1e-9:
        return None
    s = (table_z - origin[2]) / direction[2]
    return None if s <= 0 else origin + s * direction
