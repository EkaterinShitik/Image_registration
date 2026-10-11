"""Image loading, canvas alignment and tissue segmentation utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import color, exposure, filters, morphology


def load_rgb(path: str) -> np.ndarray:
    """Load an image as a float32 RGB array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert a float32 RGB array in [0, 1] to grayscale."""
    return color.rgb2gray(rgb).astype(np.float32)


def pad_to_common_canvas(images: list[np.ndarray], fill: float = 1.0, round_to: int = 32) -> list[np.ndarray]:
    """Pad a list of images (2D gray or 3D HWC) to a shared, top-left-aligned canvas.

    The two source slides have slightly different pixel dimensions
    (694x600 vs 715x600). Padding to a common canvas (rather than resizing)
    avoids introducing an extra, uncontrolled scaling deformation before the
    registration methods even run. The canvas is additionally rounded up to a
    multiple of ``round_to`` pixels per side, since the VoxelMorph U-Net
    requires dimensions divisible by 2**(number of downsampling levels).
    """
    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    h, w = max(heights), max(widths)
    if round_to > 1:
        h = int(np.ceil(h / round_to) * round_to)
        w = int(np.ceil(w / round_to) * round_to)
    out = []
    for im in images:
        if im.ndim == 2:
            canvas = np.full((h, w), fill, dtype=im.dtype)
            canvas[: im.shape[0], : im.shape[1]] = im
        else:
            canvas = np.full((h, w, im.shape[2]), fill, dtype=im.dtype)
            canvas[: im.shape[0], : im.shape[1], :] = im
        out.append(canvas)
    return out


def crop_to_mask_bbox(
    gray: np.ndarray, masks: dict[str, np.ndarray], reference: np.ndarray, margin_frac: float = 0.1
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Crop a grayscale image and its co-registered masks to `reference`'s bounding box.

    `reference` is typically a Tissue mask. The box is expanded by
    `margin_frac` of its own size on each side before cropping. Anchoring each
    slide's content near the top-left corner like this makes
    ``pad_to_common_canvas`` padding comparable in scale and position when the
    two slides were scanned with very different stage/canvas extents (e.g.
    different scanners or modalities), rather than aligning full, differently
    sized scan areas.
    """
    ys, xs = np.nonzero(reference)
    if ys.size == 0:
        return gray, masks
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    margin_y = int(round((y1 - y0) * margin_frac))
    margin_x = int(round((x1 - x0) * margin_frac))
    h, w = gray.shape
    y0 = max(0, y0 - margin_y)
    y1 = min(h, y1 + margin_y)
    x0 = max(0, x0 - margin_x)
    x1 = min(w, x1 + margin_x)
    cropped_gray = gray[y0:y1, x0:x1]
    cropped_masks = {cls: m[y0:y1, x0:x1] for cls, m in masks.items()}
    return cropped_gray, cropped_masks


def coarse_prealign(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    moving_masks: dict[str, np.ndarray],
    candidate_rotations_deg: tuple[int, ...] = (0, 90, 180, 270),
) -> tuple[np.ndarray, dict[str, np.ndarray], int, tuple[float, float]]:
    """Coarsely pre-align `moving_gray`/`moving_masks` onto `fixed_gray` before
    the non-rigid registration backends run.

    The non-rigid backends (SimpleITK B-spline, ANTs SyN, VoxelMorph) each
    start with an affine/rigid pre-alignment stage, but those have only a
    narrow capture range - they cannot recover from a large, discrete
    misalignment such as a 90-degree rotation between two independently
    mounted scans of the same slide (e.g. H&E vs. COMET). This searches the 4
    axis-aligned rotations of `moving_gray` and, for each, estimates the
    residual translation against `fixed_gray` via FFT phase correlation,
    keeping the rotation/translation combination with the lowest residual
    error - handing the fine, non-rigid registration a much smaller
    deformation left to solve.
    """
    from scipy.ndimage import shift as ndi_shift
    from skimage.registration import phase_cross_correlation

    best: tuple[float, int, tuple[float, float]] | None = None
    for degrees in candidate_rotations_deg:
        rotated = np.rot90(moving_gray, k=degrees // 90)
        fixed_padded, rotated_padded = pad_to_common_canvas([fixed_gray, rotated], round_to=1)
        shift_yx, error, _phase = phase_cross_correlation(fixed_padded, rotated_padded, upsample_factor=4)
        if best is None or error < best[0]:
            best = (error, degrees, (float(shift_yx[0]), float(shift_yx[1])))
    _, degrees, shift_yx = best

    def _apply(img: np.ndarray, *, is_mask: bool) -> np.ndarray:
        rotated = np.rot90(img, k=degrees // 90)
        order = 0 if is_mask else 1
        cval = 0.0 if is_mask else 1.0
        shifted = ndi_shift(rotated.astype(np.float32), shift=shift_yx, order=order, mode="constant", cval=cval)
        return (shifted > 0.5).astype(img.dtype) if is_mask else shifted.astype(np.float32)

    aligned_gray = _apply(moving_gray, is_mask=False)
    aligned_masks = {cls: _apply(m, is_mask=True) for cls, m in moving_masks.items()}
    return aligned_gray, aligned_masks, degrees, shift_yx


def match_histogram_to_reference(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remap `image`'s intensity distribution to match `reference`'s histogram.

    Cross-modality pairs (e.g. H&E brightfield vs. a fluorescence channel) can
    have very different brightness/contrast distributions even after polarity
    inversion, which hurts registration metrics that assume a direct intensity
    relationship (e.g. VoxelMorph's local NCC). Matching histograms before
    registration makes the two distributions comparable without altering
    spatial structure.
    """
    return exposure.match_histograms(image, reference).astype(np.float32)


def segment_tissue_mask(gray: np.ndarray) -> np.ndarray:
    """Segment the tissue (foreground) from the white slide background.

    H&E / Romanowsky-Giemsa stained slides have a bright, near-uniform white
    background, so a global Otsu threshold on grayscale intensity reliably
    separates tissue (darker, stained) from background (brighter).
    """
    threshold = filters.threshold_otsu(gray)
    mask = gray < threshold
    mask = morphology.remove_small_objects(mask, max_size=64)
    mask = morphology.remove_small_holes(mask, max_size=256)
    mask = morphology.closing(mask, morphology.disk(3))
    return mask.astype(np.uint8)
