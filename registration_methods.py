"""Three independent non-rigid registration backends: SimpleITK, ANTsPy and VoxelMorph.

Common conventions used across all three methods (see also ``metrics.py``):

- ``warped_image``: the moving image resampled into the fixed image's grid.
- ``displacement_field``: ndarray of shape (H, W, 2), ``field[row, col] =
  (d_row, d_col)``, mapping a fixed-image pixel to its corresponding location
  in the moving image (``moving_space_point = fixed_space_point + displacement``).
- ``transform_points(points_xy)``: maps (x, y) points defined in the fixed
  image into their predicted corresponding (x, y) location in the moving
  image, used for the TRE metric.
- ``warp_mask(mask)``: resamples a binary mask defined in the moving image
  into the fixed image grid (nearest-neighbor interpolation).
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class RegistrationResult:
    name: str
    warped_image: np.ndarray
    displacement_field: np.ndarray  # (H, W, 2) as (d_row, d_col)
    transform_points: Callable[[np.ndarray], np.ndarray]
    warp_mask: Callable[[np.ndarray], np.ndarray]
    runtime_sec: float = 0.0


def run_simpleitk_bspline(fixed: np.ndarray, moving: np.ndarray, grid_spacing_px: int = 40) -> RegistrationResult:
    """Non-rigid registration with SimpleITK: affine pre-alignment + multi-resolution B-spline FFD."""
    import SimpleITK as sitk

    start = time.time()
    fixed_img = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_img = sitk.GetImageFromArray(moving.astype(np.float32))

    # Stage 1: affine pre-alignment (handles any global translation/rotation/scale).
    affine_init = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetMetricSamplingStrategy(affine_reg.RANDOM)
    affine_reg.SetMetricSamplingPercentage(0.3)
    affine_reg.SetInterpolator(sitk.sitkLinear)
    affine_reg.SetInitialTransform(affine_init, inPlace=True)
    affine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200
    )
    affine_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    affine_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    affine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    affine_transform = affine_reg.Execute(fixed_img, moving_img)

    # Stage 2: non-rigid B-spline free-form deformation, initialized after the affine stage.
    mesh_size = [max(1, s // grid_spacing_px) for s in fixed_img.GetSize()]
    bspline_init = sitk.BSplineTransformInitializer(fixed_img, mesh_size, order=3)

    composite_init = sitk.CompositeTransform([affine_transform, bspline_init])

    bspline_reg = sitk.ImageRegistrationMethod()
    bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
    bspline_reg.SetMetricSamplingPercentage(0.3)
    bspline_reg.SetInterpolator(sitk.sitkLinear)
    bspline_reg.SetInitialTransform(bspline_init, inPlace=True)
    bspline_reg.SetMovingInitialTransform(affine_transform)
    bspline_reg.SetOptimizerAsLBFGSB(numberOfIterations=150)
    bspline_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    bspline_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    bspline_reg.Execute(fixed_img, moving_img)

    final_transform = sitk.CompositeTransform([affine_transform, bspline_init])
    runtime = time.time() - start

    warped = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 1.0)
    warped_arr = sitk.GetArrayFromImage(warped)

    disp_field = sitk.TransformToDisplacementField(
        final_transform,
        sitk.sitkVectorFloat64,
        fixed_img.GetSize(),
        fixed_img.GetOrigin(),
        fixed_img.GetSpacing(),
        fixed_img.GetDirection(),
    )
    disp_arr_xy = sitk.GetArrayFromImage(disp_field)  # (H, W, 2) as (d_x, d_y)
    disp_arr = disp_arr_xy[..., ::-1]  # -> (d_row, d_col)

    def transform_points(points_xy: np.ndarray) -> np.ndarray:
        return np.array([final_transform.TransformPoint((float(x), float(y))) for x, y in points_xy])

    def warp_mask(mask: np.ndarray) -> np.ndarray:
        mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
        warped_mask = sitk.Resample(mask_img, fixed_img, final_transform, sitk.sitkNearestNeighbor, 0)
        return sitk.GetArrayFromImage(warped_mask).astype(np.uint8)

    return RegistrationResult(
        name="SimpleITK (affine + B-spline FFD)",
        warped_image=warped_arr,
        displacement_field=disp_arr,
        transform_points=transform_points,
        warp_mask=warp_mask,
        runtime_sec=runtime,
    )


def run_ants_syn(
    fixed: np.ndarray,
    moving: np.ndarray,
    type_of_transform: str = "SyNRA",
    aff_iterations: tuple[int, ...] = (2100, 1200, 1200, 10),
    aff_shrink_factors: tuple[int, ...] = (6, 4, 2, 1),
    aff_smoothing_sigmas: tuple[float, ...] = (3, 2, 1, 0),
    reg_iterations: tuple[int, ...] = (40, 20, 0),
) -> RegistrationResult:
    """Non-rigid registration with ANTsPy.

    `type_of_transform` selects the linear pre-alignment staging before the
    deformable SyN stage:

    - ``"SyNRA"`` (default): explicit Rigid -> Affine -> SyN. A single affine
      optimization has a narrow capture range, which is not enough when the
      two images differ substantially in scale/aspect ratio/resolution (e.g.
      different scanners or modalities); the Rigid stage gives a wide capture
      range for rotation/translation before Affine adds anisotropic
      scale/shear - all within one coherent multi-resolution optimization.
    - ``"SyN"``: Affine -> SyN (ANTs' standard one-shot affine stage, no
      separate rigid-only stage).

    `aff_iterations`/`aff_shrink_factors`/`aff_smoothing_sigmas` control the
    multi-resolution schedule of the linear (Rigid/Affine) stages, and
    `reg_iterations` the SyN stage; all four default to ANTs' own defaults,
    but difficult cross-modal pairs (as here) often benefit from more
    iterations and/or an extra coarse resolution level.
    """
    import ants
    import pandas as pd

    start = time.time()
    fixed_ants = ants.from_numpy(fixed.astype(np.float32))
    moving_ants = ants.from_numpy(moving.astype(np.float32))

    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=type_of_transform,
        aff_iterations=aff_iterations,
        aff_shrink_factors=aff_shrink_factors,
        aff_smoothing_sigmas=aff_smoothing_sigmas,
        reg_iterations=reg_iterations,
    )
    runtime = time.time() - start

    warped_arr = reg["warpedmovout"].numpy()
    fwdtransforms = reg["fwdtransforms"]

    tmp_prefix = os.path.join(tempfile.mkdtemp(prefix="ants_compose_"), "composed_")
    composed_path = ants.apply_transforms(
        fixed=fixed_ants, moving=moving_ants, transformlist=fwdtransforms, compose=tmp_prefix
    )
    disp_arr = ants.image_read(composed_path).numpy()  # (H, W, 2) already as (d_row, d_col)

    def transform_points(points_xy: np.ndarray) -> np.ndarray:
        # ANTs' point-transform API uses array-index order (row, col) for its
        # x/y columns (verified empirically), and the *forward* transform list
        # maps fixed-space points to moving-space points (opposite of the
        # convention used for `ants.apply_transforms` on images).
        df = pd.DataFrame({"x": points_xy[:, 1], "y": points_xy[:, 0]})
        out = ants.apply_transforms_to_points(2, df, fwdtransforms)
        rows = out["x"].to_numpy()
        cols = out["y"].to_numpy()
        return np.stack([cols, rows], axis=1)

    def warp_mask(mask: np.ndarray) -> np.ndarray:
        mask_ants = ants.from_numpy(mask.astype(np.float32))
        warped_mask = ants.apply_transforms(
            fixed=fixed_ants, moving=mask_ants, transformlist=fwdtransforms, interpolator="nearestNeighbor"
        )
        return (warped_mask.numpy() > 0.5).astype(np.uint8)

    stage_names = {"SyNRA": "ANTsPy (Rigid + Affine + SyN)", "SyN": "ANTsPy (Affine + SyN)"}
    return RegistrationResult(
        name=stage_names.get(type_of_transform, f"ANTsPy ({type_of_transform})"),
        warped_image=warped_arr,
        displacement_field=disp_arr,
        transform_points=transform_points,
        warp_mask=warp_mask,
        runtime_sec=runtime,
    )


def _local_ncc_loss(fixed_t, warped_t, win: int = 9):
    """Local (windowed) normalized cross-correlation loss (to be minimized)."""
    import torch
    import torch.nn.functional as F

    sum_filt = torch.ones(1, 1, win, win, device=fixed_t.device, dtype=fixed_t.dtype)
    pad = win // 2
    win_area = float(win * win)

    i_sum = F.conv2d(fixed_t, sum_filt, padding=pad)
    j_sum = F.conv2d(warped_t, sum_filt, padding=pad)
    i2_sum = F.conv2d(fixed_t * fixed_t, sum_filt, padding=pad)
    j2_sum = F.conv2d(warped_t * warped_t, sum_filt, padding=pad)
    ij_sum = F.conv2d(fixed_t * warped_t, sum_filt, padding=pad)

    u_i = i_sum / win_area
    u_j = j_sum / win_area

    cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * win_area
    i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * win_area
    j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * win_area

    cc = (cross * cross) / (i_var * j_var + 1e-5)
    return -torch.mean(cc)


def _gradient_smoothness_loss(flow):
    """L2 smoothness penalty on the spatial gradients of the displacement field."""
    import torch

    d_row = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    d_col = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return torch.mean(d_row**2) + torch.mean(d_col**2)


def run_voxelmorph(
    fixed: np.ndarray,
    moving: np.ndarray,
    iterations: int = 400,
    lr: float = 1e-3,
    grad_weight: float = 1.0,
) -> RegistrationResult:
    """Non-rigid registration with a VoxelMorph pairwise CNN.

    No pretrained model is available for this histology modality, so the
    network is optimized directly ("instance-specific" optimization) on this
    single image pair - a standard fallback for unsupervised deep-learning
    registration methods when no training dataset exists.
    """
    os.environ.setdefault("NEURITE_BACKEND", "pytorch")
    os.environ.setdefault("VXM_BACKEND", "pytorch")
    import torch
    from scipy.ndimage import map_coordinates

    import voxelmorph as vxm

    device = "cpu"
    H, W = fixed.shape

    torch.manual_seed(0)
    fixed_t = torch.from_numpy(fixed[None, None]).float().to(device)
    moving_t = torch.from_numpy(moving[None, None]).float().to(device)

    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=(16, 32, 32, 16),
        integration_steps=5,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    model.train()
    for _ in range(iterations):
        optimizer.zero_grad()
        flow, warped_source = model(moving_t, fixed_t, return_warped_source=True)
        loss = _local_ncc_loss(fixed_t, warped_source) + grad_weight * _gradient_smoothness_loss(flow)
        loss.backward()
        optimizer.step()
    runtime = time.time() - start

    model.eval()
    with torch.no_grad():
        flow, warped_source = model(moving_t, fixed_t, return_warped_source=True)

    warped_arr = warped_source[0, 0].cpu().numpy()
    disp_arr = flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2) as (d_row, d_col)

    spatial_transformer = vxm.nn.modules.SpatialTransformer(interpolation_mode="nearest")

    def transform_points(points_xy: np.ndarray) -> np.ndarray:
        xs = points_xy[:, 0]
        ys = points_xy[:, 1]
        coords = np.vstack([ys, xs])  # sample at (row, col) = (y, x)
        d_row = map_coordinates(disp_arr[..., 0], coords, order=1, mode="nearest")
        d_col = map_coordinates(disp_arr[..., 1], coords, order=1, mode="nearest")
        return np.stack([xs + d_col, ys + d_row], axis=1)

    def warp_mask(mask: np.ndarray) -> np.ndarray:
        mask_t = torch.from_numpy(mask[None, None]).float()
        with torch.no_grad():
            warped_mask = spatial_transformer(mask_t, flow)
        return (warped_mask[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    return RegistrationResult(
        name="VoxelMorph (pairwise CNN, instance-optimized)",
        warped_image=warped_arr,
        displacement_field=disp_arr,
        transform_points=transform_points,
        warp_mask=warp_mask,
        runtime_sec=runtime,
    )
