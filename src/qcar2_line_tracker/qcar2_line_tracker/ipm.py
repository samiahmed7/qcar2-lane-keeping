#!/usr/bin/env python3
"""
Inverse Perspective Mapping (IPM) utility.

Loads a calibration YAML produced by `ipm_calibrate`, then provides:
  - `warp(frame)`        — camera frame → bird's-eye frame
  - `unwarp(frame)`      — bird's-eye frame → camera frame
  - `warp_points(pts)`   — list of (u, v) camera pixels → bird's-eye pixels
  - `unwarp_points(pts)` — bird's-eye pixels → camera pixels
  - `px_to_metres(x_px, y_px)` — bird's-eye pixel → real-world (x_m, y_m)
  - `metres_to_px(x_m, y_m)`   — real-world → bird's-eye pixel

Real-world coordinate convention (in metres, in vehicle frame):
  - x_m: distance ahead of the car (forward)
  - y_m: lateral offset (left positive, right negative)

The calibration YAML records `src_points`, `dst_points`, `warped_size`,
`m_per_px_x`, `m_per_px_y`. Everything else is derived.
"""
from pathlib import Path
import cv2
import numpy as np
import yaml


class IPM:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.config_path = config_path
        self.src = np.float32(cfg['src_points'])
        self.dst = np.float32(cfg['dst_points'])
        self.warped_w, self.warped_h = cfg['warped_size']
        self.m_per_px_x = float(cfg['m_per_px_x'])
        self.m_per_px_y = float(cfg['m_per_px_y'])
        self.real_width_m = float(cfg.get('real_width_m', 0.0))
        self.real_length_m = float(cfg.get('real_length_m', 0.0))

        # Use saved M/Minv if present, else recompute
        if 'M' in cfg and 'Minv' in cfg:
            self.M = np.array(cfg['M'], dtype=np.float64)
            self.Minv = np.array(cfg['Minv'], dtype=np.float64)
        else:
            self.M = cv2.getPerspectiveTransform(self.src, self.dst)
            self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    # ────────────────────────────────────────────────────────────
    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            frame, self.M, (self.warped_w, self.warped_h),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, frame: np.ndarray, target_size=None) -> np.ndarray:
        if target_size is None:
            target_size = (frame.shape[1], frame.shape[0])
        return cv2.warpPerspective(
            frame, self.Minv, target_size, flags=cv2.INTER_LINEAR
        )

    def warp_points(self, pts: np.ndarray) -> np.ndarray:
        """Camera-image pixel(s) → bird's-eye pixel(s)."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.Minv).reshape(-1, 2)

    # ────────────────────────────────────────────────────────────
    def px_to_metres(self, x_px: float, y_px: float) -> tuple:
        """
        Bird's-eye pixel (u, v) → vehicle-frame metres (x_m forward, y_m left).

        Convention used in the calibration YAML:
        - Bottom of the bird's-eye image (v = warped_h-1) is closest to the car
        - Centre of the image (u = warped_w/2) is dead ahead
        """
        x_m = (self.warped_h - 1 - y_px) * self.m_per_px_y
        y_m = (self.warped_w / 2.0 - x_px) * self.m_per_px_x  # +y_m = left
        return float(x_m), float(y_m)

    def metres_to_px(self, x_m: float, y_m: float) -> tuple:
        u = self.warped_w / 2.0 - y_m / self.m_per_px_x
        v = self.warped_h - 1 - x_m / self.m_per_px_y
        return float(u), float(v)


def default_config_path() -> str:
    """Look up the installed config file path."""
    from ament_index_python.packages import get_package_share_directory
    return str(Path(get_package_share_directory('qcar2_line_tracker'))
               / 'config' / 'ipm.yaml')
