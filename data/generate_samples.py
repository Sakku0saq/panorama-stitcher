#!/usr/bin/env python3
"""
generate_samples.py
--------------------
Generates a set of synthetic, textured, overlapping sample images so the
panorama pipeline can be tested end-to-end without needing external photos.

A large textured "scene" image is created (random shapes + noise for
strong SIFT keypoints), then split into overlapping horizontal crops
that simulate a panning camera.

Run:
    python data/generate_samples.py
"""

import os
import numpy as np
import cv2


def generate_scene(width: int = 1600, height: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = np.full((height, width, 3), 40, dtype=np.uint8)

    # Textured noise background for feature-richness.
    noise = rng.integers(0, 60, (height, width, 3), dtype=np.uint8)
    scene = cv2.add(scene, noise)

    # Scatter shapes across the whole width so overlapping crops share features.
    for _ in range(120):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        color = tuple(int(c) for c in rng.integers(50, 255, 3))
        shape_type = rng.integers(0, 3)
        if shape_type == 0:
            radius = int(rng.integers(8, 25))
            cv2.circle(scene, (x, y), radius, color, -1)
        elif shape_type == 1:
            size = int(rng.integers(15, 40))
            cv2.rectangle(scene, (x, y), (x + size, y + size), color, -1)
        else:
            pts = np.array(
                [[x, y], [x + int(rng.integers(10, 30)), y + int(rng.integers(10, 30))],
                 [x - int(rng.integers(10, 30)), y + int(rng.integers(10, 30))]],
                dtype=np.int32,
            )
            cv2.fillPoly(scene, [pts], color)

    return scene


def split_overlapping(scene: np.ndarray, num_splits: int = 3, overlap_frac: float = 0.35):
    h, w = scene.shape[:2]
    piece_w = int(w / (num_splits - (num_splits - 1) * overlap_frac))
    step = int(piece_w * (1 - overlap_frac))

    crops = []
    x = 0
    for i in range(num_splits):
        end = min(x + piece_w, w)
        crops.append(scene[:, x:end].copy())
        if i < num_splits - 1:
            x += step
    return crops


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
    os.makedirs(out_dir, exist_ok=True)

    scene = generate_scene()
    crops = split_overlapping(scene, num_splits=3, overlap_frac=0.35)

    for idx, crop in enumerate(crops, start=1):
        path = os.path.join(out_dir, f"sample_{idx}.jpg")
        cv2.imwrite(path, crop)
        print(f"Wrote {path}  shape={crop.shape}")


if __name__ == "__main__":
    main()
