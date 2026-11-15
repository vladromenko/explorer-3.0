# Copied verbatim from so101_autonomous_pick_place/src/so101_pick_place/geometry/table_projection.py (2026-11-16).
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ProjectedPoint:
    xyz_m: tuple[float, float, float]
    uncertainty_m: float


class TableProjection:
    def __init__(self, camera_matrix: np.ndarray, distortion: np.ndarray, camera_from_robot: np.ndarray, table_z_m: float = 0.0, reprojection_rmse_px: float = 0.0) -> None:
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        self.distortion = np.asarray(distortion, dtype=float)
        self.camera_from_robot = np.asarray(camera_from_robot, dtype=float)
        self.robot_from_camera = np.linalg.inv(self.camera_from_robot)
        self.table_z_m = float(table_z_m)
        self.reprojection_rmse_px = float(reprojection_rmse_px)

    def pixel_to_table(self, pixel: tuple[float, float]) -> ProjectedPoint:
        points = np.array([[[pixel[0], pixel[1]]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.distortion)[0, 0]
        ray_camera = np.array([undistorted[0], undistorted[1], 1.0])
        rotation = self.robot_from_camera[:3, :3]
        origin = self.robot_from_camera[:3, 3]
        ray_robot = rotation @ ray_camera
        if abs(ray_robot[2]) < 1e-8:
            raise ValueError("viewing ray is parallel to table")
        scale = (self.table_z_m - origin[2]) / ray_robot[2]
        if scale <= 0:
            raise ValueError("table intersection is behind camera")
        point = origin + scale * ray_robot
        focal = max(float(self.camera_matrix[0, 0]), 1.0)
        uncertainty = max(0.002, abs(scale) * max(self.reprojection_rmse_px, 1.0) / focal)
        return ProjectedPoint(tuple(float(v) for v in point), uncertainty)

    def table_to_pixel(self, xyz_m: tuple[float, float, float]) -> tuple[float, float]:
        point = np.asarray(xyz_m, dtype=float).reshape(1, 1, 3)
        rotation, _ = cv2.Rodrigues(self.camera_from_robot[:3, :3])
        translation = self.camera_from_robot[:3, 3]
        projected, _ = cv2.projectPoints(point, rotation, translation, self.camera_matrix, self.distortion)
        value = projected[0, 0]
        return float(value[0]), float(value[1])

