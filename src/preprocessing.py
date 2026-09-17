"""
preprocessing.py
-----------------
Module 1: Preprocessing.

Responsible for loading, validating, and preparing input images before
feature extraction. Keeping this as its own module means the rest of
the pipeline never has to deal with raw file I/O or malformed images.
"""

import os
from typing import List

import cv2
import numpy as np

from src.logger_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class ImageLoadError(Exception):
    """Raised when an input image cannot be found or read."""


def load_images(paths: List[str]) -> List[np.ndarray]:
    """
    Load a list of images from disk with validation.

    Parameters
    ----------
    paths : List[str]
        File paths to the input images, in left-to-right (or capture) order.

    Returns
    -------
    List[np.ndarray]
        Loaded BGR images.

    Raises
    ------
    ImageLoadError
        If a path does not exist, has an unsupported extension, or
        cannot be decoded by OpenCV.
    """
    if len(paths) < 2:
        raise ImageLoadError(
            f"At least 2 images are required to stitch a panorama, got {len(paths)}."
        )

    images = []
    for path in paths:
        if not os.path.isfile(path):
            raise ImageLoadError(f"Image not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(
                f"Unsupported file extension '{ext}' for {path}. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        img = cv2.imread(path)
        if img is None:
            raise ImageLoadError(f"OpenCV failed to decode image: {path}")

        images.append(img)
        logger.debug("Loaded image %s with shape %s", path, img.shape)

    logger.info("Successfully loaded %d images.", len(images))
    return images


def resize_if_needed(image: np.ndarray, max_dimension: int = 1000) -> np.ndarray:
    """
    Downscale an image if its largest dimension exceeds max_dimension.

    This directly supports the Performance non-functional requirement:
    feature detection and matching are O(n^2)-ish in the number of
    keypoints, so capping resolution keeps runtime predictable on
    large photos without materially hurting stitching quality.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    max_dimension : int
        Maximum allowed width or height, in pixels.

    Returns
    -------
    np.ndarray
        Resized image (or the original, if already small enough).
    """
    h, w = image.shape[:2]
    largest = max(h, w)

    if largest <= max_dimension:
        return image

    scale = max_dimension / float(largest)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    logger.debug("Resized image from %s to %s", (w, h), new_size)
    return resized


def preprocess_images(paths: List[str], max_dimension: int = 1000) -> List[np.ndarray]:
    """
    Full preprocessing pipeline: load + validate + resize.

    Parameters
    ----------
    paths : List[str]
        Paths to input images.
    max_dimension : int
        Max width/height passed to resize_if_needed.

    Returns
    -------
    List[np.ndarray]
        Preprocessed images ready for feature detection.
    """
    images = load_images(paths)
    return [resize_if_needed(img, max_dimension) for img in images]
