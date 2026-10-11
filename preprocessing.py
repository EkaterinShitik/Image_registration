"""Image loading, canvas alignment and tissue segmentation utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import color, filters, morphology


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
