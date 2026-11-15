"""Mechanics copied from ~/Projects/so101_autonomous_pick_place (SO-101 pick-and-place project).

Provenance: copied 2026-11-16 at the user's request («скопируй механику ... в эксплорер»); the source
project and Wristlab are not modified and stay runnable. Copied: overview-camera table projection
(geometry/table_projection.py, verbatim) and the validated overview hand-eye calibration
(config/calibration.json -> configs/so101_station/overview_calibration.json, reprojection 2.0 px,
independent physical control 3.1 px). If the camera or the robot base moves, this calibration is
invalid: the station then refuses to measure.
"""
