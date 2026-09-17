"""Unit tests for src/stitching.py"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stitching import warp_and_stitch, _crop_black_borders


def test_warp_and_stitch_identity_homography_same_size():
    img_a = np.full((100, 100, 3), 200, dtype=np.uint8)
    img_b = np.full((100, 100, 3), 100, dtype=np.uint8)
    H = np.eye(3, dtype=np.float64)

    result = warp_and_stitch(img_a, img_b, H)
    assert result.ndim == 3
    assert result.shape[2] == 3
    # With identical placement, overlap should blend to something between the two values.
    assert result.size > 0


def test_crop_black_borders_removes_empty_padding():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas[20:80, 30:70] = 255
    cropped = _crop_black_borders(canvas)
    assert cropped.shape[0] <= 60
    assert cropped.shape[1] <= 40


def test_crop_black_borders_all_black_returns_original():
    canvas = np.zeros((50, 50, 3), dtype=np.uint8)
    cropped = _crop_black_borders(canvas)
    assert cropped.shape == canvas.shape
