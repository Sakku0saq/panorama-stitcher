"""Unit tests for src/feature_detection.py"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_detection import (
    detect_and_describe,
    match_descriptors,
    InsufficientFeaturesError,
)


def _textured_image(seed=0, size=(300, 300)):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    for _ in range(30):
        x, y = int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))
        cv2.circle(img, (x, y), int(rng.integers(5, 20)), (255, 0, 0), -1)
    return img


def test_detect_and_describe_on_textured_image():
    img = _textured_image()
    keypoints, descriptors = detect_and_describe(img)
    assert len(keypoints) >= 4
    assert descriptors.shape[0] == len(keypoints)


def test_detect_and_describe_flat_image_raises():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    with pytest.raises(InsufficientFeaturesError):
        detect_and_describe(flat)


def test_match_descriptors_between_identical_images():
    img = _textured_image(seed=1)
    kp1, desc1 = detect_and_describe(img)
    kp2, desc2 = detect_and_describe(img)
    matches = match_descriptors(desc1, desc2)
    assert len(matches) >= 4


def test_match_descriptors_too_few_raises():
    desc_a = np.random.rand(5, 128).astype(np.float32)
    desc_b = np.random.rand(5, 128).astype(np.float32)
    with pytest.raises(InsufficientFeaturesError):
        match_descriptors(desc_a, desc_b, ratio_thresh=0.01)
