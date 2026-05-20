#!/usr/bin/env python3
"""
Lane-line detector for the bird's-eye view.

Inputs : a BGR bird's-eye-warped image of the road.
Outputs: pixel positions of the LEFT and RIGHT lane lines, sampled at
         several y-rows of the image, plus a debug overlay.

Algorithm — sliding window:
    1. White-mask the image (HSV: high V, low S).
    2. Take a histogram of white-pixel columns in the bottom third of the
       image. Find the dominant peak in the left half and right half →
       these are the starting x positions of the left and right lines.
    3. Walk N windows up the image. Each window: collect the white pixels
       inside it, recentre the window on their mean x if there are enough,
       record the centre.
    4. Return the per-window centres (one polyline per side) and the
       overlay image with windows drawn.

The output centres are in BIRD'S-EYE PIXEL coords. The caller converts to
metres via the IPM scale factors.
"""
import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        n_windows: int = 9,
        window_margin: int = 50,
        min_pixels_recenter: int = 30,
        min_pixels_per_side: int = 100,
        white_v_low: int = 180,
        white_s_high: int = 60,
    ):
        self.n_windows = n_windows
        self.window_margin = window_margin
        self.min_pixels_recenter = min_pixels_recenter
        self.min_pixels_per_side = min_pixels_per_side
        self.white_v_low = white_v_low
        self.white_s_high = white_s_high

    # ────────────────────────────────────────────────────────────
    def detect(self, bev_bgr: np.ndarray,
               seed_left=None, seed_right=None):
        """
        Run sliding-window detection on the bird's-eye-view image.

        Parameters
        ----------
        bev_bgr     : the bird's-eye warped BGR image
        seed_left   : optional starting x (pixels) for the LEFT line tracker.
                      If None, picked from a histogram of the bottom third.
        seed_right  : same for the RIGHT line tracker.

        Returns
        -------
        result : dict with keys
            'mask'            HxW uint8 binary mask of white pixels
            'left_centres'    list of (x, y) pixel tuples, sampled bottom-up
            'right_centres'   list of (x, y) pixel tuples
            'left_count'      total white pixels collected for left line
            'right_count'     total ditto for right line
            'overlay'         BGR debug visualisation with windows + centres
            'left_base'       starting x used (None if no seed available)
            'right_base'      starting x used
        """
        h, w = bev_bgr.shape[:2]

        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0,   0, self.white_v_low], dtype=np.uint8),
            np.array([180, self.white_s_high, 255], dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        # Starting x for each side. If the caller supplied a seed (from a
        # higher-level state estimator), trust that — it knows where the
        # lines should be, not just "whatever's in this half of the image".
        # Otherwise fall back to a histogram of the bottom third.
        if seed_left is not None and seed_right is not None:
            left_base = int(np.clip(seed_left, 0, w - 1))
            right_base = int(np.clip(seed_right, 0, w - 1))
        else:
            bottom = mask[h * 2 // 3:, :]
            hist = np.sum(bottom > 0, axis=0)
            mid = w // 2
            left_base = int(np.argmax(hist[:mid]))
            right_base = int(np.argmax(hist[mid:]) + mid)
            if hist[left_base] < 8:
                left_base = None
            if hist[right_base] < 8:
                right_base = None

        # Sliding-window line tracking
        window_h = h // self.n_windows
        nz_y, nz_x = mask.nonzero()
        left_centres, right_centres = [], []
        left_pix_idx, right_pix_idx = [], []

        left_x = left_base
        right_x = right_base

        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Tint the overlay slightly toward the real image so the user can see road
        overlay = cv2.addWeighted(overlay, 0.6, bev_bgr, 0.4, 0)

        for win in range(self.n_windows):
            y_hi = h - win * window_h
            y_lo = h - (win + 1) * window_h
            y_mid = (y_hi + y_lo) // 2

            for side, current, centres, pix_idx in (
                ('L', left_x,  left_centres,  left_pix_idx),
                ('R', right_x, right_centres, right_pix_idx),
            ):
                if current is None:
                    continue
                x_lo = current - self.window_margin
                x_hi = current + self.window_margin
                colour = (255, 255, 0) if side == 'L' else (0, 255, 255)
                cv2.rectangle(overlay, (x_lo, y_lo), (x_hi, y_hi), colour, 1)

                inside = ((nz_y >= y_lo) & (nz_y < y_hi) &
                          (nz_x >= x_lo) & (nz_x < x_hi))
                idx = np.where(inside)[0]
                pix_idx.append(idx)
                if len(idx) >= self.min_pixels_recenter:
                    new_x = int(np.mean(nz_x[idx]))
                    centres.append((new_x, y_mid))
                    if side == 'L':
                        left_x = new_x
                    else:
                        right_x = new_x
                # else: don't drop window, just don't recentre

        left_count = int(sum(len(i) for i in left_pix_idx))
        right_count = int(sum(len(i) for i in right_pix_idx))

        # Draw the detected centres on the overlay
        for x, y in left_centres:
            cv2.circle(overlay, (x, y), 4, (255, 80, 0), -1)
        for x, y in right_centres:
            cv2.circle(overlay, (x, y), 4, (0, 80, 255), -1)

        return {
            'mask': mask,
            'left_centres': left_centres,
            'right_centres': right_centres,
            'left_count': left_count,
            'right_count': right_count,
            'left_base': left_base,
            'right_base': right_base,
            'overlay': overlay,
        }
