"""
homography.py
--------------
Module 3: Homography Estimation.

Implements RANSAC-based homography estimation between two sets of
matched keypoints (Module 2 of the CV syllabus: Camera & Epipolar
Geometry, Homography, RANSAC).
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.logger_config import get_logger

logger = get_logger(__name__)


class HomographyEstimationError(Exception):
    """Raised when a valid homography cannot be estimated."""


def compute_homography(
    keypoints_a: List[cv2.KeyPoint],
    keypoints_b: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_reproj_thresh: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the homography mapping image B's coordinates into image A's
    coordinate frame, using RANSAC for robustness to outlier matches.

    Parameters
    ----------
    keypoints_a, keypoints_b : List[cv2.KeyPoint]
        Keypoints from image A (reference) and image B (to be warped).
    matches : List[cv2.DMatch]
        Good matches between the two descriptor sets (queryIdx -> A, trainIdx -> B).
    ransac_reproj_thresh : float
        Max reprojection error (px) for a point pair to be treated as an inlier.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (3x3 homography matrix, inlier mask)

    Raises
    ------
    HomographyEstimationError
        If RANSAC fails to find a valid homography.
    """
    pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, ransac_reproj_thresh)

    if H is None:
        raise HomographyEstimationError(
            "cv2.findHomography returned None; matches may be degenerate "
            "(e.g. all collinear) or too noisy."
        )

    inlier_count = int(mask.sum()) if mask is not None else 0
    logger.debug(
        "Homography estimated with %d/%d inliers.", inlier_count, len(matches)
    )

    if inlier_count < 4:
        raise HomographyEstimationError(
            f"Only {inlier_count} RANSAC inliers; homography is unreliable."
        )

    return H, mask
