"""Registration quality metrics for the benchmark.

All displacement fields passed to these functions are expected to use the
array-axis convention ``field[row, col] = (d_row, d_col)`` - i.e. the last
axis matches the order of the first two array axes (row/y first, col/x
second). Each registration backend in ``registration_methods.py`` is
responsible for converting to this common convention.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from skimage.metrics import normalized_mutual_information


def nmi_score(fixed: np.ndarray, warped: np.ndarray, bins: int = 32) -> float:
    """Normalized Mutual Information between the fixed and warped images."""
    return float(normalized_mutual_information(fixed, warped, bins=bins))


def tre_score(ground_truth_pts: np.ndarray, predicted_pts: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Target Registration Error: Euclidean distance (px) between corresponding points.

    Returns (mean, std, per-point distances).
    """
    dists = np.linalg.norm(predicted_pts - ground_truth_pts, axis=1)
    return float(dists.mean()), float(dists.std()), dists


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice Similarity Coefficient between two binary tissue masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    intersection = np.logical_and(a, b).sum()
    return float(2.0 * intersection / denom)


def jacobian_determinant_stats(displacement_field: np.ndarray) -> dict:
    """Statistics of the Jacobian determinant of a 2D displacement field.

    Parameters
    ----------
    displacement_field : ndarray, shape (H, W, 2)
        Displacement ``(d_row, d_col)`` at every pixel of the fixed image grid,
        mapping fixed-image coordinates to moving-image coordinates.

    Returns
    -------
    dict with the mean/std of det(J) and the percentage of pixels with a
    non-positive determinant (folding / non-diffeomorphic regions - lower is
    better, 0% means a fully plausible, invertible deformation).
    """
    d_row = displacement_field[..., 0]
    d_col = displacement_field[..., 1]

    d_row_drow, d_row_dcol = np.gradient(d_row)
    d_col_drow, d_col_dcol = np.gradient(d_col)

    jac_00 = 1.0 + d_row_drow
    jac_01 = d_row_dcol
    jac_10 = d_col_drow
    jac_11 = 1.0 + d_col_dcol

    det = jac_00 * jac_11 - jac_01 * jac_10
    return {
        "jac_mean": float(np.mean(det)),
        "jac_std": float(np.std(det)),
        "jac_negative_pct": float(np.mean(det <= 0) * 100.0),
    }


def _mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Extract (x, y) boundary/contour pixel coordinates of a binary mask."""
    contours = measure.find_contours(mask.astype(float), level=0.5)
    if not contours:
        return np.empty((0, 2))
    points = np.concatenate(contours, axis=0)  # (row, col)
    return points[:, ::-1]  # -> (x, y)


def hausdorff_95(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff Distance (HD95, in px) between two tissue contours."""
    pts_a = _mask_boundary_points(mask_a)
    pts_b = _mask_boundary_points(mask_b)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan")

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    d_ab, _ = tree_b.query(pts_a)
    d_ba, _ = tree_a.query(pts_b)

    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
