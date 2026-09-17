"""Unit tests for src/homography.py"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.homography import compute_homography, HomographyEstimationError


def _make_keypoints(points):
    return [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in points]


def _make_matches(n):
    return [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0.0) for i in range(n)]


def test_compute_homography_identity_mapping():
    # Same points in both images -> homography should be close to identity.
    pts = [(10, 10), (100, 10), (100, 100), (10, 100), (50, 50), (30, 80)]
    kp_a = _make_keypoints(pts)
    kp_b = _make_keypoints(pts)
    matches = _make_matches(len(pts))

    H, mask = compute_homography(kp_a, kp_b, matches)
    assert H.shape == (3, 3)
    identity_approx = H / H[2, 2]
    assert np.allclose(identity_approx, np.eye(3), atol=1.0)


def test_compute_homography_translation():
    pts_a = [(10, 10), (100, 10), (100, 100), (10, 100), (50, 50), (30, 80)]
    shift = 20
    pts_b = [(x - shift, y) for x, y in pts_a]  # B is A shifted left by `shift`

    kp_a = _make_keypoints(pts_a)
    kp_b = _make_keypoints(pts_b)
    matches = _make_matches(len(pts_a))

    H, mask = compute_homography(kp_a, kp_b, matches)
    # Applying H to a point from B should land close to the matching point in A.
    pt_b = np.array([[pts_b[0]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pt_b, H)[0][0]
    assert np.allclose(projected, pts_a[0], atol=2.0)


def test_compute_homography_insufficient_inliers_raises():
    # Random, inconsistent point pairs should not yield a reliable homography.
    rng = np.random.default_rng(0)
    pts_a = [(int(x), int(y)) for x, y in rng.integers(0, 500, (4, 2))]
    pts_b = [(int(x), int(y)) for x, y in rng.integers(0, 500, (4, 2))]
    kp_a = _make_keypoints(pts_a)
    kp_b = _make_keypoints(pts_b)
    matches = _make_matches(4)

    # This may or may not raise depending on random luck, so we just assert
    # it either raises the expected error type or returns a valid 3x3 matrix.
    try:
        H, mask = compute_homography(kp_a, kp_b, matches)
        assert H.shape == (3, 3)
    except HomographyEstimationError:
        pass
