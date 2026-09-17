"""
stitching.py
------------
Module 4: Stitching & Blending.

Warps images into a common canvas using the estimated homography and
blends overlapping regions to produce the final panorama.
"""

from typing import Tuple

import cv2
import numpy as np

from src.logger_config import get_logger

logger = get_logger(__name__)


def warp_and_stitch(
    image_a: np.ndarray, image_b: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """
    Warp image_b into image_a's frame using homography H, place image_a
    on the resulting canvas, and linearly feather-blend the overlap
    region so the seam is not visible as a hard edge.

    Parameters
    ----------
    image_a : np.ndarray
        Reference (left) image.
    image_b : np.ndarray
        Image to be warped (right), mapped via H into image_a's frame.
    H : np.ndarray
        3x3 homography mapping image_b's coordinates into image_a's frame.

    Returns
    -------
    np.ndarray
        Stitched panorama (BGR), cropped to remove empty black borders.
    """
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    # Compute the canvas size needed to hold both warped images.
    corners_b = np.float32(
        [[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]
    ).reshape(-1, 1, 2)
    warped_corners_b = cv2.perspectiveTransform(corners_b, H)

    corners_a = np.float32(
        [[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]
    ).reshape(-1, 1, 2)

    all_corners = np.concatenate((corners_a, warped_corners_b), axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64
    )

    canvas_size = (x_max - x_min, y_max - y_min)

    warped_b = cv2.warpPerspective(image_b, translation @ H, canvas_size)

    canvas_a = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    canvas_a[-y_min : -y_min + h_a, -x_min : -x_min + w_a] = image_a

    panorama = _feather_blend(canvas_a, warped_b)
    cropped = _crop_black_borders(panorama)

    logger.info("Stitched panorama with final size %s.", cropped.shape[:2])
    return cropped


def _feather_blend(canvas_a: np.ndarray, warped_b: np.ndarray) -> np.ndarray:
    """
    Blend two same-sized canvases using a distance-transform-based
    feather mask, so overlapping regions fade smoothly rather than
    showing a hard cut.
    """
    mask_a = (cv2.cvtColor(canvas_a, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    mask_b = (cv2.cvtColor(warped_b, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    overlap = (mask_a & mask_b).astype(bool)

    if not overlap.any():
        # No overlap detected: simple addition (shouldn't normally happen
        # for images that were successfully matched).
        return cv2.add(canvas_a, warped_b)

    dist_a = cv2.distanceTransform(mask_a * 255, cv2.DIST_L2, 5).astype(np.float64)
    dist_b = cv2.distanceTransform(mask_b * 255, cv2.DIST_L2, 5).astype(np.float64)

    denom = dist_a + dist_b
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(denom > 0, dist_a / (denom + 1e-6), 0.5)
    alpha = alpha[..., None]

    blended = canvas_a.astype(np.float32) * (1 - alpha) + warped_b.astype(
        np.float32
    ) * alpha

    result = canvas_a.copy()
    result[mask_b.astype(bool) & ~mask_a.astype(bool)] = warped_b[
        mask_b.astype(bool) & ~mask_a.astype(bool)
    ]
    result[overlap] = blended[overlap].astype(np.uint8)

    return result


def _crop_black_borders(image: np.ndarray) -> np.ndarray:
    """Trim fully-black rows/columns left over from warping onto a larger canvas."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y : y + h, x : x + w]
