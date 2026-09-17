"""
pipeline.py
-----------
Orchestrates the full panorama stitching workflow by chaining the four
functional modules together:

    preprocessing -> feature_detection -> homography -> stitching

For N input images, images are stitched sequentially: the running
panorama is treated as the "left" image and each subsequent image is
matched and merged into it in order.
"""

from typing import List

import cv2
import numpy as np

from src.logger_config import get_logger
from src.preprocessing import preprocess_images, ImageLoadError
from src.feature_detection import (
    detect_and_describe,
    match_descriptors,
    InsufficientFeaturesError,
)
from src.homography import compute_homography, HomographyEstimationError
from src.stitching import warp_and_stitch

logger = get_logger(__name__)


class PipelineError(Exception):
    """Top-level error wrapping any stage failure, for a clean CLI message."""


def run_pipeline(
    image_paths: List[str],
    max_dimension: int = 1000,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 4.0,
) -> np.ndarray:
    """
    Run the complete stitching pipeline over an ordered list of image paths.

    Parameters
    ----------
    image_paths : List[str]
        Paths to input images, left-to-right in capture order.
    max_dimension : int
        Max width/height for preprocessing resize.
    ratio_thresh : float
        Lowe's ratio test threshold for feature matching.
    ransac_reproj_thresh : float
        RANSAC reprojection error threshold (pixels) for homography.

    Returns
    -------
    np.ndarray
        Final stitched panorama image (BGR).

    Raises
    ------
    PipelineError
        Wraps any failure from the underlying stages with a clear message.
    """
    try:
        logger.info("Stage 1/4: Preprocessing %d images.", len(image_paths))
        images = preprocess_images(image_paths, max_dimension=max_dimension)

        panorama = images[0]

        for i in range(1, len(images)):
            logger.info("Stitching image %d/%d into panorama.", i + 1, len(images))

            logger.info("Stage 2/4: Detecting & matching features.")
            kp_pano, desc_pano = detect_and_describe(panorama)
            kp_next, desc_next = detect_and_describe(images[i])
            matches = match_descriptors(desc_pano, desc_next, ratio_thresh)

            logger.info("Stage 3/4: Estimating homography via RANSAC.")
            H, _ = compute_homography(
                kp_pano, kp_next, matches, ransac_reproj_thresh
            )

            logger.info("Stage 4/4: Warping and blending.")
            panorama = warp_and_stitch(panorama, images[i], H)

        logger.info("Pipeline completed successfully.")
        return panorama

    except ImageLoadError as e:
        logger.error("Preprocessing failed: %s", e)
        raise PipelineError(f"Preprocessing error: {e}") from e
    except InsufficientFeaturesError as e:
        logger.error("Feature detection/matching failed: %s", e)
        raise PipelineError(f"Feature matching error: {e}") from e
    except HomographyEstimationError as e:
        logger.error("Homography estimation failed: %s", e)
        raise PipelineError(f"Homography error: {e}") from e
    except Exception as e:  # noqa: BLE001 - top-level safety net, re-raised with context
        logger.exception("Unexpected error in pipeline.")
        raise PipelineError(f"Unexpected error: {e}") from e
