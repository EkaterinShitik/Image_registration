"""Automatic landmark correspondence detection between fixed and moving images.

There is no manually annotated ground truth for this image pair, so we use
robust ORB feature matching + RANSAC (affine) to obtain a set of point
correspondences between the two tissue sections. These points are treated as
a proxy ground truth for the Target Registration Error (TRE) metric only -
none of the registration methods themselves ever see these points, so they
remain an independent evaluation signal.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform


def detect_matched_landmarks(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    n_keypoints: int = 1000,
    min_samples: int = 3,
    residual_threshold: float = 4.0,
    max_landmarks: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect robust point correspondences between two images via ORB + RANSAC.

    Returns
    -------
    fixed_pts, moving_pts : ndarray of shape (N, 2)
        Inlier correspondences as (x, y) pixel coordinates, i.e.
        ``fixed_pts[i]`` in the fixed image corresponds to the same physical
        tissue location as ``moving_pts[i]`` in the moving image.
    """
    orb_fixed = ORB(n_keypoints=n_keypoints)
    orb_moving = ORB(n_keypoints=n_keypoints)
    orb_fixed.detect_and_extract(fixed_gray)
    orb_moving.detect_and_extract(moving_gray)

    matches = match_descriptors(orb_fixed.descriptors, orb_moving.descriptors, cross_check=True)
    if len(matches) < min_samples:
        raise RuntimeError("Not enough ORB matches to estimate landmark correspondences.")

    fixed_kp = orb_fixed.keypoints[matches[:, 0]]  # (row, col)
    moving_kp = orb_moving.keypoints[matches[:, 1]]

    model, inliers = ransac(
        (fixed_kp, moving_kp),
        AffineTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=2000,
        rng=seed,
    )
    if inliers is None or inliers.sum() < min_samples:
        raise RuntimeError("Could not find enough robust landmark correspondences.")

    fixed_kp = fixed_kp[inliers]
    moving_kp = moving_kp[inliers]

    if len(fixed_kp) > max_landmarks:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(fixed_kp), size=max_landmarks, replace=False)
        fixed_kp = fixed_kp[idx]
        moving_kp = moving_kp[idx]

    # (row, col) -> (x, y)
    fixed_pts = fixed_kp[:, ::-1].astype(np.float64)
    moving_pts = moving_kp[:, ::-1].astype(np.float64)
    return fixed_pts, moving_pts
