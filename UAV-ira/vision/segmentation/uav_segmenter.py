# This file is the segmenter for the auv

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

@dataclass
class SegmenterConfig: # we're doing green and white since the football field will be green with white lines
    green_lo: Tuple[int, int, int] = (35, 40, 40)
    green_hi: Tuple[int, int, int] = (90, 255, 255)

    white_lo: Tuple[int, int, int] = (0, 0, 180)
    white_hi: Tuple[int, int, int] = (179, 60, 255)

    bg_kernel: int = 5
    obstacle_kernel: int = 7
    bg_open_iter: int = 1
    bg_close_iter: int = 2
    ob_open_iter: int = 1
    ob_close_iter: int = 2

    grid_h: int = 40
    grid_w: int = 40
    occ_thresh: float = 0.03

    process_interval_sec: float = 0.5 # this is the "still snap every .5 secs"

    def _odd(k: int) -> int:
        return k if k % 2 == 1 else k + 1


class FieldObstacleSegmenter:
    # This is the segmenter using opencv for the uav and assumes the camera will face 
    # directly dowwnwards to the football field

    def __init__(self, cfg: SegmenterConfig = SegmenterConfig()):
        self.cfg = cfg
        self._last_run = 0.0

    #sees if enough time has passed to process another frame
    def should_process_now(self) -> bool:
        now = time.time()
        return (now - self._last_run) >= self.cfg.process_interval_sec

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty or None")
        
        self._last_run = time.time()

        #converting to hsv color space for better color segmentation
        #hsv is a color system that matches how humans see them (hue, saturation, value)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        #this is where it will detect the green and white (not objects)
        green = cv2.inRange(
            hsv, np.array(self.cfg.green_lo, np.uint8), np.array(self.cfg.green_hi, np.uint8)
        )
        white = cv2.inRange(
            hsv, np.array(self.cfg.white_lo, np.uint8), np.array(self.cfg.white_hi, np.uint8)
        )

        background = cv2.bitwise_or(green, white)

        bgk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(self.cfg.bg_kernel), _odd(self.cfg.bg_kernel)))
        if self.cfg.bg_open_iter > 0:
            background = cv2.morphologyEx(background, cv2.MORPH_OPEN, bgk, iterations=self.cfg.bg_open_iter)
        if self.cfg.bg_close_iter > 0:
            background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, bgk, iterations=self.cfg.bg_close_iter)

        # the obstacle is the inverse of the background mask, since we want to find non-green/white areas
        obstacle = cv2.bitwise_not(background)

        obk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(self.cfg.obstacle_kernel), _odd(self.cfg.obstacle_kernel))
        )
        if self.cfg.ob_open_iter > 0:
            obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_OPEN, obk, iterations=self.cfg.ob_open_iter)
        if self.cfg.ob_close_iter > 0:
            obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_CLOSE, obk, iterations=self.cfg.ob_close_iter)

        # now we have the obstacle mask, we can convert it to a grid representation
        grid = self._mask_to_grid(obstacle)

        return obstacle, grid
    
        # helper function to convert the obstacle mask to a grid representation, where each cell is marked as occupied if enough pixels in that cell are marked as obstacles in the mask
    def _mask_to_grid(self, obstacle_mask: np.ndarray) -> np.ndarray:
        H, W = obstacle_mask.shape[:2]
        gh, gw = self.cfg.grid_h, self.cfg.grid_w

        cell_h = H // gh
        cell_w = W // gw
        if cell_h <= 0 or cell_w <= 0:
            raise ValueError(f"Grid too large for image: image=({H},{W}) grid=({gh},{gw})")

        if obstacle_mask.ndim == 3:
            obstacle_mask = obstacle_mask[:, :, 0]

        grid = np.zeros((gh, gw), dtype=np.uint8)
        for r in range(gh):
            for c in range(gw):
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                cell = obstacle_mask[y0:y1, x0:x1]
                occ = float(np.mean(cell > 0))
                grid[r, c] = 1 if occ >= self.cfg.occ_thresh else 0

        return grid
    
    # helper function to overlay the obstacle mask on the original frame for visualization, where the obstacles will be highlighted in red with some transparency
    @staticmethod
    def overlay(frame_bgr: np.ndarray, obstacle_mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        overlay = frame_bgr.copy()
        red = np.zeros_like(frame_bgr)
        red[:, :, 2] = 255
        m3 = cv2.cvtColor(obstacle_mask, cv2.COLOR_GRAY2BGR) if obstacle_mask.ndim == 2 else obstacle_mask
        return np.where(m3 > 0, (alpha * red + (1 - alpha) * overlay).astype(np.uint8), overlay)
    