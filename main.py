#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point for the Panorama Stitcher.

Usage
-----
    python main.py --images img1.jpg img2.jpg img3.jpg --output output/panorama.jpg

Run `python main.py --help` for the full list of options.
"""

import argparse
import sys
import time

import cv2

from src.logger_config import get_logger
from src.pipeline import run_pipeline, PipelineError

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panorama-stitcher",
        description=(
            "Stitch 2 or more overlapping images into a single panorama "
            "using SIFT feature matching and RANSAC homography estimation."
        ),
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Paths to input images, in left-to-right order (2 or more required).",
    )
    parser.add_argument(
        "--output",
        default="output/panorama.jpg",
        metavar="PATH",
        help="Path to save the stitched panorama (default: output/panorama.jpg).",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1000,
        help="Downscale images so their largest side is at most this many pixels "
        "(default: 1000). Lower this for faster runs on large photos.",
    )
    parser.add_argument(
        "--ratio-thresh",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold for feature matching (default: 0.75).",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=4.0,
        help="RANSAC reprojection error threshold in pixels (default: 4.0).",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if len(args.images) < 2:
        parser.error("At least 2 images are required (--images img1 img2 ...).")

    start = time.time()
    logger.info("Starting panorama stitching for %d images.", len(args.images))

    try:
        panorama = run_pipeline(
            image_paths=args.images,
            max_dimension=args.max_dimension,
            ratio_thresh=args.ratio_thresh,
            ransac_reproj_thresh=args.ransac_thresh,
        )
    except PipelineError as e:
        logger.error("Stitching failed: %s", e)
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        return 1

    ok = cv2.imwrite(args.output, panorama)
    if not ok:
        logger.error("Failed to write output image to %s", args.output)
        print(f"\n[ERROR] Could not write output to {args.output}\n", file=sys.stderr)
        return 1

    elapsed = time.time() - start
    logger.info("Done in %.2f seconds. Panorama saved to %s", elapsed, args.output)
    print(f"\nPanorama saved to: {args.output}  ({elapsed:.2f}s)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
