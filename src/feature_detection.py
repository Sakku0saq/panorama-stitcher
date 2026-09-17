"""
feature_detection.py
---------------------
Module 2: Feature Detection & Matching.

Implements SIFT keypoint detection and descriptor computation
(Module 3 of the CV syllabus: Feature Extraction), followed by
descriptor matching with Lowe's ratio test to discard ambiguous
matches before homography estimation.
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.logger_config import get_logger

logger = get_logger(__name__)


class InsufficientFeaturesError(Exception):
    """Raised when too few keypoints/matches are found to proceed reliably."""


def detect_and_describe(
    image: np.ndarray,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect SIFT keypoints and compute their descriptors.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.

    Returns
    -------
    Tuple[List[cv2.KeyPoint], np.ndarray]
        Keypoints and their corresponding 128-d SIFT descriptors.

    Raises
    ------
    InsufficientFeaturesError
        If fewer than 4 keypoints are found (4 correspondences are the
        theoretical minimum required to estimate a homography).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 4:
        raise InsufficientFeaturesError(
            f"Only {len(keypoints) if keypoints else 0} keypoints found; "
            "need at least 4 to estimate a homography. Try a higher-texture image."
        )

    logger.debug("Detected %d SIFT keypoints.", len(keypoints))
    return keypoints, descriptors


def match_descriptors(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio_thresh: float = 0.75,
) -> List[cv2.DMatch]:
    """
    Match two descriptor sets using a brute-force matcher with
    Lowe's ratio test to filter out ambiguous matches.

    Parameters
    ----------
    descriptors_a, descriptors_b : np.ndarray
        SIFT descriptors from two images.
    ratio_thresh : float
        Lowe's ratio test threshold (lower = stricter).

    Returns
    -------
    List[cv2.DMatch]
        Filtered "good" matches.

    Raises
    ------
    InsufficientFeaturesError
        If fewer than 4 good matches survive the ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(descriptors_a, descriptors_b, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise InsufficientFeaturesError(
            f"Only {len(good_matches)} good matches after ratio test; "
            "need at least 4 for RANSAC homography. Images may not overlap enough."
        )

    logger.debug("Found %d good matches after ratio test.", len(good_matches))
    return good_matches
