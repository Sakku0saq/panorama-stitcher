"""Unit tests for src/preprocessing.py"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_images, resize_if_needed, preprocess_images, ImageLoadError


@pytest.fixture
def tmp_images(tmp_path):
    paths = []
    for i in range(2):
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        path = str(tmp_path / f"img{i}.jpg")
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


def test_load_images_success(tmp_images):
    images = load_images(tmp_images)
    assert len(images) == 2
    assert all(isinstance(img, np.ndarray) for img in images)


def test_load_images_too_few_raises():
    with pytest.raises(ImageLoadError):
        load_images(["only_one.jpg"])


def test_load_images_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        load_images([str(tmp_path / "does_not_exist.jpg"), str(tmp_path / "also_missing.jpg")])


def test_load_images_bad_extension_raises(tmp_path):
    bad_file = tmp_path / "notanimage.txt"
    bad_file.write_text("hello")
    good_file = tmp_path / "img.jpg"
    cv2.imwrite(str(good_file), np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ImageLoadError):
        load_images([str(bad_file), str(good_file)])


def test_resize_if_needed_downscales_large_image():
    large = np.zeros((2000, 3000, 3), dtype=np.uint8)
    resized = resize_if_needed(large, max_dimension=1000)
    assert max(resized.shape[:2]) <= 1000


def test_resize_if_needed_leaves_small_image_unchanged():
    small = np.zeros((200, 300, 3), dtype=np.uint8)
    resized = resize_if_needed(small, max_dimension=1000)
    assert resized.shape == small.shape


def test_preprocess_images_end_to_end(tmp_images):
    result = preprocess_images(tmp_images, max_dimension=1000)
    assert len(result) == 2
