"""Transfer QuPath Tissue/Nucleus masks from an H&E whole-slide image (WSI) onto
the *same* slide re-imaged with the COMET multiplex-fluorescence method, via
non-rigid registration, and compare the transferred masks against the real
masks independently computed on the COMET image.

Unlike ``main_mask.py`` (adjacent serial H&E sections, same scanner/modality),
this pairs two scans of the *same* physical slide acquired with different
instruments:

- H&E (moving): ``NZ 40X Cukierman H&E 1007775 _ 205938.ndpi`` - brightfield
  RGB, read directly with ``tiffslide``.
- COMET (fixed): ``20260223_..._Tonsil.ome.tiff`` - a pyramidal, multichannel
  (DAPI/TRITC/CD4-Cy5) UINT16 OME-TIFF written by Bio-Formats. ``tiffslide``
  can open it (it is a generic tiled pyramid), but ``read_region`` must be
  requested as a raw array (``as_array=True``) since PIL cannot represent
  multichannel UINT16 data; we then take the DAPI channel, percentile-stretch
  it to [0, 1] and invert it so nuclei are dark on a bright background,
  matching the H&E polarity the registration backends were tuned against.

Prerequisite (one-time, inside QuPath, "Run for project"):

- ``export_annotations.groovy`` - exports each image's Tissue annotation.
- ``export_detections.groovy`` - exports each image's nucleus detections
  (e.g. StarDist), used as the "Nucleus" mask (all detections count, since
  they are not further classified).

Both scripts can also be run headlessly, e.g.:
``QuPath script qupath_deveral_slides_alignment/scripts/export_detections.groovy
-p qupath_deveral_slides_alignment/project.qpproj -i "<image name>"``.

Pipeline: read each slide at a matched physical resolution (microns/pixel) ->
rasterize Tissue/Nucleus masks from the GeoJSON exports -> register H&E ->
COMET (SimpleITK B-spline, ANTsPy SyN, VoxelMorph) -> warp the H&E masks into
the COMET grid and score them against COMET's own masks (Dice/IoU/
Precision/Recall/HD95), saving an overlay per method/class.
"""

from __future__ import annotations

import argparse
import functools
import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

from main_mask import (
    METHOD_FUNCS,
    _default_annotation_path,
    _geometry_to_polygons,
    _make_overlay,
    _pad_to_shape,
    evaluate_mask_transfer,
    identity_result,
    load_annotation_polygons,
    rasterize_mask,
    read_wsi_level,
)
from preprocessing import coarse_prealign, crop_to_mask_bbox, match_histogram_to_reference, pad_to_common_canvas, rgb_to_gray
from registration_methods import run_ants_syn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
QUPATH_DIR = os.path.join(ROOT_DIR, "qupath_deveral_slides_alignment")
ANNOTATIONS_DIR = os.path.join(QUPATH_DIR, "annotations")
NUCLEI_DIR = os.path.join(ANNOTATIONS_DIR, "nuclei")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

DEFAULT_HE_NDPI = os.path.join(DATA_DIR, "NZ 40X Cukierman H&E 1007775 _ 205938.ndpi")
DEFAULT_COMET_TIFF = os.path.join(DATA_DIR, "20260223_143249_4_2mIA6M_SIO_PB_(3plex,all Bodor)_Tonsil.ome.tiff")

DEFAULT_TARGET_MPP = 8.0  # shared physical resolution (microns/pixel) used for registration
DEFAULT_CROP_MARGIN = 0.1  # fraction of the Tissue bbox size added as margin before cropping
DEFAULT_COMET_CHANNEL = 0  # DAPI


def _slide_mpp(path: str) -> float:
    """Microns-per-pixel at full resolution, as reported by tiffslide."""
    import tiffslide

    with tiffslide.TiffSlide(path) as slide:
        mpp = slide.properties.get("tiffslide.mpp-x")
    if mpp is None:
        raise ValueError(f"Could not determine microns-per-pixel for {path}")
    return float(mpp)


def _slide_dimensions(path: str) -> tuple[int, int]:
    """(width, height) at full resolution, as reported by tiffslide."""
    import tiffslide

    with tiffslide.TiffSlide(path) as slide:
        return slide.dimensions


def _downsample_for_target_mpp(path: str, target_mpp: float) -> float:
    """Exact (non-pyramid-snapped) downsample relative to level-0 pixels that
    achieves `target_mpp` microns/pixel for this slide's own calibration."""
    return max(1.0, target_mpp / _slide_mpp(path))


def _target_shape_for_downsample(path: str, downsample: float) -> tuple[int, int]:
    width, height = _slide_dimensions(path)
    return max(1, round(height / downsample)), max(1, round(width / downsample))


def _resample_to_exact_shape(gray: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if gray.shape == target_shape:
        return gray
    return resize(gray, target_shape, order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )


def read_he_level(ndpi_path: str, target_mpp: float) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Read the H&E slide and resample it to exactly `target_mpp` microns/pixel.

    Pyramid levels are snapped to each file's own downsample factors (usually
    powers of two), which generally don't line up between two different
    scanners/modalities. The nearest level is read first for I/O efficiency,
    then explicitly resized (bilinear) to the exact shape implied by
    `target_mpp`, so the H&E and COMET grids - and the masks rasterized at the
    same exact downsample - share one common physical pixel spacing.

    Returns ``(gray, downsample, raw_gray, raw_downsample)``, where
    ``raw_gray``/``raw_downsample`` are the grayscale image and downsample
    *before* this resampling step, for "where we started" diagnostics.
    """
    downsample = _downsample_for_target_mpp(ndpi_path, target_mpp)
    rgb, raw_downsample = read_wsi_level(ndpi_path, downsample)
    raw_gray = rgb_to_gray(rgb)
    gray = _resample_to_exact_shape(raw_gray, _target_shape_for_downsample(ndpi_path, downsample))
    return gray, downsample, raw_gray, raw_downsample


def read_comet_level(
    tiff_path: str, target_mpp: float, channel: int = DEFAULT_COMET_CHANNEL, invert: bool = True
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Read one channel (default: DAPI) of the COMET fluorescence OME-TIFF and
    resample it to exactly `target_mpp` microns/pixel (see `read_he_level`).

    Percentile-stretches the channel to [0, 1] and, by default, inverts it so
    nuclei appear dark on a bright background - the same polarity as H&E.

    Returns ``(gray, downsample, raw_gray, raw_downsample)``, where
    ``raw_gray``/``raw_downsample`` are the percentile-stretched but
    *not inverted, not resampled* channel and its native pyramid downsample,
    for "where we started" diagnostics.
    """
    import tiffslide

    downsample = _downsample_for_target_mpp(tiff_path, target_mpp)
    with tiffslide.TiffSlide(tiff_path) as slide:
        level = slide.get_best_level_for_downsample(downsample)
        raw_downsample = slide.level_downsamples[level]
        size = slide.level_dimensions[level]
        arr = slide.read_region((0, 0), level, size, as_array=True)

    channel_arr = arr[..., channel].astype(np.float32)
    lo, hi = np.percentile(channel_arr, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    raw_gray = np.clip((channel_arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    gray = (1.0 - raw_gray) if invert else raw_gray

    gray = _resample_to_exact_shape(gray, _target_shape_for_downsample(tiff_path, downsample))
    return gray, downsample, raw_gray, raw_downsample


def _save_grayscale(path: str, gray: np.ndarray) -> None:
    Image.fromarray((np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def rasterize_detection_mask(geojson_path: str, shape: tuple[int, int], downsample: float) -> np.ndarray:
    """Rasterize a QuPath detection export (e.g. StarDist nuclei) into a binary mask.

    Detection exports can contain hundreds of thousands of objects (hundreds of
    MB of JSON), so features are streamed one at a time with ``ijson`` instead
    of loading the whole file into memory.
    """
    import ijson
    from PIL import ImageDraw

    h, w = shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    with open(geojson_path, "rb") as f:
        for feature in ijson.items(f, "item", use_float=True):
            for exterior, holes in _geometry_to_polygons(feature["geometry"]):
                draw.polygon([(x / downsample, y / downsample) for x, y in exterior], outline=1, fill=1)
                for hole in holes:
                    draw.polygon([(x / downsample, y / downsample) for x, y in hole], outline=0, fill=0)
    return (np.asarray(img) > 0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--he-ndpi", default=DEFAULT_HE_NDPI, help="H&E slide (moving).")
    parser.add_argument("--comet-tiff", default=DEFAULT_COMET_TIFF, help="COMET OME-TIFF slide (fixed).")
    parser.add_argument("--he-tissue", default=None, help="GeoJSON Tissue export for --he-ndpi (default: auto-detected).")
    parser.add_argument("--comet-tissue", default=None, help="GeoJSON Tissue export for --comet-tiff (default: auto-detected).")
    parser.add_argument("--he-nuclei", default=None, help="GeoJSON detection export for --he-ndpi (default: auto-detected).")
    parser.add_argument("--comet-nuclei", default=None, help="GeoJSON detection export for --comet-tiff (default: auto-detected).")
    parser.add_argument("--target-mpp", type=float, default=DEFAULT_TARGET_MPP, help="Shared physical resolution (microns/pixel) for registration.")
    parser.add_argument("--crop-margin", type=float, default=DEFAULT_CROP_MARGIN, help="Crop each slide to its Tissue bounding box + this fraction of margin before registration (0 to disable, e.g. --crop-margin 0).")
    parser.add_argument("--comet-channel", type=int, default=DEFAULT_COMET_CHANNEL, help="COMET channel index to register on (default: 0 = DAPI).")
    parser.add_argument("--no-invert-comet", action="store_true", help="Don't invert the COMET channel before registration.")
    parser.add_argument("--no-match-histograms", action="store_true", help="Don't remap the H&E histogram to match COMET's before registration.")
    parser.add_argument("--no-coarse-prealign", action="store_true", help="Skip the coarse 90-degree rotation + residual translation pre-alignment step before registration.")
    parser.add_argument("--classes", default="Tissue,Nucleus", help="Comma-separated subset of: Tissue, Nucleus.")
    parser.add_argument("--methods", default="simpleitk,ants,voxelmorph", help="Comma-separated subset of: " + ", ".join(METHOD_FUNCS))
    parser.add_argument(
        "--ants-transform",
        default="SyNRA",
        choices=["SyNRA", "SyN"],
        help="ANTs linear pre-alignment staging before SyN: SyNRA = Rigid->Affine->SyN, SyN = Affine->SyN. Only affects the 'ants' method.",
    )
    parser.add_argument("--ants-aff-iterations", default="2100,1200,1200,10", help="Comma-separated affine iterations per resolution level, coarse->fine (ANTs default shown). Only affects the 'ants' method.")
    parser.add_argument("--ants-aff-shrink-factors", default="6,4,2,1", help="Comma-separated affine shrink factors per resolution level, coarse->fine. Only affects the 'ants' method.")
    parser.add_argument("--ants-aff-smoothing-sigmas", default="3,2,1,0", help="Comma-separated affine smoothing sigmas per resolution level, coarse->fine. Only affects the 'ants' method.")
    parser.add_argument("--ants-syn-iterations", default="40,20,0", help="Comma-separated SyN iterations per resolution level, coarse->fine. Only affects the 'ants' method.")
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    he_tissue_path = args.he_tissue or _default_annotation_path(args.he_ndpi, ANNOTATIONS_DIR)
    comet_tissue_path = args.comet_tissue or _default_annotation_path(args.comet_tiff, ANNOTATIONS_DIR)
    he_nuclei_path = args.he_nuclei or _default_annotation_path(args.he_ndpi, NUCLEI_DIR)
    comet_nuclei_path = args.comet_nuclei or _default_annotation_path(args.comet_tiff, NUCLEI_DIR)

    for label, path in {
        "H&E tissue annotations": he_tissue_path,
        "COMET tissue annotations": comet_tissue_path,
        "H&E nuclei detections": he_nuclei_path,
        "COMET nuclei detections": comet_nuclei_path,
    }.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "Export QuPath annotations/detections to GeoJSON first by running "
                "export_annotations.groovy and export_detections.groovy ('Run for "
                "project') inside QuPath (see qupath_deveral_slides_alignment/scripts/)."
            )

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_FUNCS:
            raise ValueError(f"Unknown method '{m}', choose from {list(METHOD_FUNCS)}")

    overlay_dir = os.path.join(args.output_dir, "he_comet_mask_transfer_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"Loading H&E slide (moving, target mpp~{args.target_mpp}): {args.he_ndpi}")
    moving_gray, moving_downsample, moving_raw_gray, moving_raw_downsample = read_he_level(args.he_ndpi, args.target_mpp)
    print(f"Loading COMET slide (fixed, channel {args.comet_channel}, target mpp~{args.target_mpp}): {args.comet_tiff}")
    fixed_gray, fixed_downsample, fixed_raw_gray, fixed_raw_downsample = read_comet_level(
        args.comet_tiff, args.target_mpp, args.comet_channel, invert=not args.no_invert_comet
    )

    print("Loading QuPath Tissue annotation masks...")
    moving_tissue = load_annotation_polygons(he_tissue_path).get("Tissue", [])
    fixed_tissue = load_annotation_polygons(comet_tissue_path).get("Tissue", [])
    moving_masks: dict[str, np.ndarray] = {"Tissue": rasterize_mask(moving_tissue, moving_gray.shape, moving_downsample)}
    fixed_masks: dict[str, np.ndarray] = {"Tissue": rasterize_mask(fixed_tissue, fixed_gray.shape, fixed_downsample)}

    if not args.no_invert_comet:
        print("Forcing COMET background (outside Tissue) to white, matching H&E's brightfield background...")
        fixed_gray = np.where(fixed_masks["Tissue"].astype(bool), fixed_gray, 1.0).astype(np.float32)

    if "Nucleus" in classes:
        print("Loading QuPath nucleus detection masks (this can take a moment)...")
        moving_masks["Nucleus"] = rasterize_detection_mask(he_nuclei_path, moving_gray.shape, moving_downsample)
        fixed_masks["Nucleus"] = rasterize_detection_mask(comet_nuclei_path, fixed_gray.shape, fixed_downsample)

    print("Saving 'before preprocessing' snapshots (native resolution, pre-scaling/pre-inversion)...")
    initial_dir = os.path.join(overlay_dir, "initial_raw")
    os.makedirs(initial_dir, exist_ok=True)
    _save_grayscale(os.path.join(initial_dir, "HE_raw_grayscale.png"), moving_raw_gray)
    _save_grayscale(os.path.join(initial_dir, "COMET_raw_grayscale_noninverted.png"), fixed_raw_gray)
    _save_grayscale(os.path.join(initial_dir, "HE_processed_grayscale.png"), moving_gray)
    _save_grayscale(os.path.join(initial_dir, "COMET_processed_grayscale_inverted.png"), fixed_gray)

    moving_raw_masks = {"Tissue": rasterize_mask(moving_tissue, moving_raw_gray.shape, moving_raw_downsample)}
    fixed_raw_masks = {"Tissue": rasterize_mask(fixed_tissue, fixed_raw_gray.shape, fixed_raw_downsample)}
    if "Nucleus" in classes:
        moving_raw_masks["Nucleus"] = rasterize_detection_mask(he_nuclei_path, moving_raw_gray.shape, moving_raw_downsample)
        fixed_raw_masks["Nucleus"] = rasterize_detection_mask(comet_nuclei_path, fixed_raw_gray.shape, fixed_raw_downsample)
    for cls in classes:
        h = max(fixed_raw_masks[cls].shape[0], moving_raw_masks[cls].shape[0])
        w = max(fixed_raw_masks[cls].shape[1], moving_raw_masks[cls].shape[1])
        overlay = _make_overlay(_pad_to_shape(fixed_raw_masks[cls], h, w), _pad_to_shape(moving_raw_masks[cls], h, w))
        overlay.save(os.path.join(initial_dir, f"{cls}_initial_raw_overlay.png"))

    if args.crop_margin > 0:
        print(f"Cropping each slide to its Tissue bounding box (+{args.crop_margin:.0%} margin)...")
        moving_gray, moving_masks = crop_to_mask_bbox(moving_gray, moving_masks, moving_masks["Tissue"], args.crop_margin)
        fixed_gray, fixed_masks = crop_to_mask_bbox(fixed_gray, fixed_masks, fixed_masks["Tissue"], args.crop_margin)

    if not args.no_coarse_prealign:
        print("Coarse pre-alignment: searching 90-degree rotations + residual translation...")
        moving_gray, moving_masks, rotation_deg, shift_yx = coarse_prealign(fixed_gray, moving_gray, moving_masks)
        print(f"  chosen rotation: {rotation_deg} deg, residual shift (row, col): ({shift_yx[0]:.1f}, {shift_yx[1]:.1f})")

    if not args.no_match_histograms:
        print("Matching H&E histogram to COMET's brightness distribution...")
        moving_gray = match_histogram_to_reference(moving_gray, fixed_gray)

    print("Padding to a common canvas...")
    fixed_gray, moving_gray = pad_to_common_canvas([fixed_gray, moving_gray])
    height, width = fixed_gray.shape
    moving_masks = {cls: _pad_to_shape(m, height, width) for cls, m in moving_masks.items()}
    fixed_masks = {cls: _pad_to_shape(m, height, width) for cls, m in fixed_masks.items()}

    # Registration backends require single-channel input; guard against any
    # accidental 3-channel array reaching this point.
    if moving_gray.ndim == 3:
        moving_gray = rgb_to_gray(moving_gray)
    if fixed_gray.ndim == 3:
        fixed_gray = rgb_to_gray(fixed_gray)

    print("Running registration methods (this can take a while)...")
    method_funcs = dict(METHOD_FUNCS)
    if "ants" in methods:
        method_funcs["ants"] = functools.partial(
            run_ants_syn,
            type_of_transform=args.ants_transform,
            aff_iterations=tuple(int(v) for v in args.ants_aff_iterations.split(",")),
            aff_shrink_factors=tuple(int(v) for v in args.ants_aff_shrink_factors.split(",")),
            aff_smoothing_sigmas=tuple(float(v) for v in args.ants_aff_smoothing_sigmas.split(",")),
            reg_iterations=tuple(int(v) for v in args.ants_syn_iterations.split(",")),
        )
        print(f"  (ants: type_of_transform={args.ants_transform}, aff_iterations={args.ants_aff_iterations}, "
              f"aff_shrink_factors={args.ants_aff_shrink_factors}, aff_smoothing_sigmas={args.ants_aff_smoothing_sigmas}, "
              f"reg_iterations={args.ants_syn_iterations})")
    results = [identity_result(moving_gray)]
    for m in methods:
        print(f"  -> {m}")
        results.append(method_funcs[m](fixed_gray, moving_gray))

    rows = []
    for result in results:
        for cls in classes:
            row, warped_mask = evaluate_mask_transfer(result, moving_masks[cls], fixed_masks[cls])
            row["class"] = cls
            rows.append(row)

            overlay = _make_overlay(fixed_masks[cls], warped_mask)
            safe_method = result.name.replace("/", "_")
            overlay.save(os.path.join(overlay_dir, f"{cls}_{safe_method}.png"))

    df = pd.DataFrame(rows)[["class", "method", "dice", "iou", "precision", "recall", "hd95", "runtime_sec"]]

    csv_path = os.path.join(args.output_dir, "he_comet_mask_transfer_metrics.csv")
    df.to_csv(csv_path, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== H&E -> COMET mask transfer results ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved metrics table -> {csv_path}")
    print(f"Saved overlays -> {overlay_dir}")


if __name__ == "__main__":
    main()
