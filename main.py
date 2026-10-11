"""Benchmark of non-rigid image registration quality for two consecutive
Romanowsky-Giemsa stained tissue sections (108.jpeg -> 109.jpeg).

Runs three independent registration backends - SimpleITK (B-spline FFD),
ANTsPy (SyN diffeomorphic) and VoxelMorph (instance-optimized CNN) - and
scores each with:

- Normalized Mutual Information (NMI)
- Target Registration Error (TRE), based on ORB-matched landmark points
- Dice Similarity Coefficient (DSC) of the segmented tissue mask
- Jacobian determinant of the deformation field (% folding pixels)
- 95th-percentile Hausdorff Distance (HD95) of the tissue contour

All metrics are collected into results/benchmark_metrics.csv and plotted
together in results/benchmark_summary.png.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from landmarks import detect_matched_landmarks
from metrics import dice_score, hausdorff_95, jacobian_determinant_stats, nmi_score, tre_score
from preprocessing import load_rgb, pad_to_common_canvas, rgb_to_gray, segment_tissue_mask
from registration_methods import RegistrationResult, run_ants_syn, run_simpleitk_bspline, run_voxelmorph
from visualize import plot_benchmark

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

FIXED_PATH = os.path.join(DATA_DIR, "108.jpeg")
MOVING_PATH = os.path.join(DATA_DIR, "109.jpeg")


def make_baseline_result(moving_gray: np.ndarray) -> RegistrationResult:
    """Identity 'registration' used as the unregistered baseline for comparison."""
    h, w = moving_gray.shape
    return RegistrationResult(
        name="Initial (unregistered)",
        warped_image=moving_gray,
        displacement_field=np.zeros((h, w, 2), dtype=np.float64),
        transform_points=lambda pts: pts.copy(),
        warp_mask=lambda mask: mask,
        runtime_sec=0.0,
    )


def evaluate(
    result: RegistrationResult,
    fixed_gray: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_pts: np.ndarray,
    moving_pts: np.ndarray,
    moving_mask: np.ndarray,
) -> dict:
    warped_mask = result.warp_mask(moving_mask)
    predicted_moving_pts = result.transform_points(fixed_pts)

    tre_mean, tre_std, _ = tre_score(moving_pts, predicted_moving_pts)
    jac_stats = jacobian_determinant_stats(result.displacement_field)

    return {
        "method": result.name,
        "nmi": nmi_score(fixed_gray, result.warped_image),
        "tre_mean": tre_mean,
        "tre_std": tre_std,
        "dice": dice_score(fixed_mask, warped_mask),
        "hd95": hausdorff_95(fixed_mask, warped_mask),
        "jac_mean": jac_stats["jac_mean"],
        "jac_negative_pct": jac_stats["jac_negative_pct"],
        "runtime_sec": result.runtime_sec,
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading {FIXED_PATH} (fixed) and {MOVING_PATH} (moving)...")
    fixed_rgb, moving_rgb = pad_to_common_canvas([load_rgb(FIXED_PATH), load_rgb(MOVING_PATH)])
    fixed_gray = rgb_to_gray(fixed_rgb)
    moving_gray = rgb_to_gray(moving_rgb)

    print("Segmenting tissue masks (Otsu threshold)...")
    fixed_mask = segment_tissue_mask(fixed_gray)
    moving_mask = segment_tissue_mask(moving_gray)

    print("Detecting ORB landmark correspondences for TRE evaluation...")
    fixed_pts, moving_pts = detect_matched_landmarks(fixed_gray, moving_gray)
    print(f"  -> {len(fixed_pts)} robust landmark correspondences found.")

    print("Running registration methods (this can take a while)...")
    results = [make_baseline_result(moving_gray)]

    print("  [1/3] SimpleITK (affine + B-spline FFD)...")
    results.append(run_simpleitk_bspline(fixed_gray, moving_gray))

    print("  [2/3] ANTsPy (SyN diffeomorphic)...")
    results.append(run_ants_syn(fixed_gray, moving_gray))

    print("  [3/3] VoxelMorph (pairwise CNN, instance-optimized)...")
    results.append(run_voxelmorph(fixed_gray, moving_gray))

    print("Computing metrics...")
    rows = [
        evaluate(result, fixed_gray, fixed_mask, fixed_pts, moving_pts, moving_mask) for result in results
    ]
    df = pd.DataFrame(rows)

    csv_path = os.path.join(RESULTS_DIR, "benchmark_metrics.csv")
    df.to_csv(csv_path, index=False)

    fig_path = os.path.join(RESULTS_DIR, "benchmark_summary.png")
    plot_benchmark(df, fig_path)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Benchmark results ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved metrics table -> {csv_path}")
    print(f"Saved summary figure -> {fig_path}")


if __name__ == "__main__":
    main()
