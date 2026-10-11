"""Transfer a QuPath pixel-classifier mask from one whole-slide image (WSI) to
an adjacent serial section, via non-rigid registration, and compare the
transferred mask against the real mask independently computed on the target
slide.

Prerequisite (one-time, inside QuPath): export each slide's annotations
(Tissue / Blood / ...) as GeoJSON with
``qupath_deveral_slides_alignment/export_annotations.groovy`` ("Run for
project"). This writes one ``<slide-name>.geojson`` file per image into
``qupath_deveral_slides_alignment/annotations/``, which this script reads.

Pipeline:

1. Read the moving and fixed ``.ndpi`` slides at a low-resolution pyramid
   level (full resolution is far too large to register directly).
2. Rasterize the GeoJSON annotation polygons for the requested
   classification(s) (default: Tissue, Blood) into binary masks at that same
   resolution.
3. Register moving -> fixed (SimpleITK B-spline, ANTsPy SyN, VoxelMorph -
   reusing ``registration_methods.py``).
4. Warp the moving slide's mask into the fixed slide's coordinate space with
   each method's ``warp_mask`` and compare it against the fixed slide's own
   (real) mask using Dice / IoU / Precision / Recall / HD95.

Requires the ``tiffslide`` package to read ``.ndpi`` pyramids (pip-installable,
no system OpenSlide library needed).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from metrics import dice_score, hausdorff_95
from preprocessing import pad_to_common_canvas, rgb_to_gray
from registration_methods import RegistrationResult, run_ants_syn, run_simpleitk_bspline, run_voxelmorph

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
QUPATH_DIR = os.path.join(ROOT_DIR, "qupath_deveral_slides_alignment")
ANNOTATIONS_DIR = os.path.join(QUPATH_DIR, "annotations")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

DEFAULT_MOVING_NDPI = os.path.join(DATA_DIR, "NZ 40X Cukierman H&E 25-1318A-1-002.ndpi")
DEFAULT_FIXED_NDPI = os.path.join(DATA_DIR, "NZ 40X Cukierman H&E 25-1318A-2-001.ndpi")
DEFAULT_DOWNSAMPLE = 32.0
DEFAULT_CLASSES = ["Tissue", "Blood"]

METHOD_FUNCS = {
    "simpleitk": run_simpleitk_bspline,
    "ants": run_ants_syn,
    "voxelmorph": run_voxelmorph,
}


def read_wsi_level(ndpi_path: str, downsample: float) -> tuple[np.ndarray, float]:
    """Read a whole-slide image at the pyramid level closest to `downsample`.

    Returns (rgb float32 in [0, 1], actual_downsample).
    """
    import tiffslide

    with tiffslide.TiffSlide(ndpi_path) as slide:
        level = slide.get_best_level_for_downsample(downsample)
        actual_downsample = slide.level_downsamples[level]
        size = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, size).convert("RGB")
    rgb = np.asarray(region, dtype=np.float32) / 255.0
    return rgb, actual_downsample


def load_annotation_polygons(geojson_path: str) -> dict[str, list[tuple[list, list]]]:
    """Load a QuPath GeoJSON export, grouped by classification name.

    Returns {classification_name: [(exterior_ring, [hole_rings...]), ...]},
    with ring coordinates in level-0 (full resolution) pixel units.
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data["features"] if isinstance(data, dict) and "features" in data else data

    by_class: dict[str, list[tuple[list, list]]] = {}
    for feature in features:
        props = feature.get("properties", {}) or {}
        classification = props.get("classification", {}) or {}
        class_name = classification.get("name", "Unclassified")
        by_class.setdefault(class_name, []).extend(_geometry_to_polygons(feature["geometry"]))
    return by_class


def _geometry_to_polygons(geometry: dict) -> list[tuple[list, list]]:
    """Normalize a GeoJSON Polygon/MultiPolygon into a list of (exterior, holes) rings."""
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    if gtype == "Polygon":
        return [(coords[0], coords[1:])]
    if gtype == "MultiPolygon":
        return [(poly[0], poly[1:]) for poly in coords]
    return []


def rasterize_mask(polygons: list[tuple[list, list]], shape: tuple[int, int], downsample: float) -> np.ndarray:
    """Rasterize (exterior, holes) polygon rings (level-0 px) into a binary mask at `downsample`."""
    h, w = shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for exterior, holes in polygons:
        draw.polygon([(x / downsample, y / downsample) for x, y in exterior], outline=1, fill=1)
        for hole in holes:
            draw.polygon([(x / downsample, y / downsample) for x, y in hole], outline=0, fill=0)
    return (np.asarray(img) > 0).astype(np.uint8)


def _pad_to_shape(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=mask.dtype)
    h, w = mask.shape
    canvas[:h, :w] = mask
    return canvas


def _default_annotation_path(ndpi_path: str) -> str:
    stem = os.path.splitext(os.path.basename(ndpi_path))[0]
    exact = os.path.join(ANNOTATIONS_DIR, f"{stem}.geojson")
    if os.path.exists(exact) or not os.path.isdir(ANNOTATIONS_DIR):
        return exact

    # QuPath's stored image name can differ from the on-disk .ndpi filename
    # (e.g. a trailing "-002" scan/rescan suffix), so fall back to the
    # longest exported GeoJSON stem that is a prefix of the .ndpi stem.
    candidates = [
        f
        for f in os.listdir(ANNOTATIONS_DIR)
        if f.endswith(".geojson") and not f.startswith("TEST ") and stem.startswith(os.path.splitext(f)[0])
    ]
    if candidates:
        return os.path.join(ANNOTATIONS_DIR, max(candidates, key=len))
    return exact


def identity_result(moving_gray: np.ndarray) -> RegistrationResult:
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


def evaluate_mask_transfer(result: RegistrationResult, moving_mask: np.ndarray, fixed_mask: np.ndarray) -> tuple[dict, np.ndarray]:
    warped_mask = result.warp_mask(moving_mask)
    tp = int(np.logical_and(warped_mask, fixed_mask).sum())
    fp = int(np.logical_and(warped_mask, np.logical_not(fixed_mask)).sum())
    fn = int(np.logical_and(np.logical_not(warped_mask), fixed_mask).sum())
    union = tp + fp + fn
    row = {
        "method": result.name,
        "dice": dice_score(fixed_mask, warped_mask),
        "iou": tp / union if union else 1.0,
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
        "hd95": hausdorff_95(fixed_mask, warped_mask),
        "runtime_sec": result.runtime_sec,
    }
    return row, warped_mask


def _make_overlay(fixed_mask: np.ndarray, warped_mask: np.ndarray) -> Image.Image:
    """RGB overlay: green = correctly transferred, red = missed, blue = spurious."""
    h, w = fixed_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[np.logical_and(fixed_mask, warped_mask)] = (0, 200, 0)
    rgb[np.logical_and(fixed_mask, np.logical_not(warped_mask))] = (220, 0, 0)
    rgb[np.logical_and(np.logical_not(fixed_mask), warped_mask)] = (0, 0, 220)
    return Image.fromarray(rgb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moving-ndpi", default=DEFAULT_MOVING_NDPI, help="Slide whose mask is transferred (source).")
    parser.add_argument("--fixed-ndpi", default=DEFAULT_FIXED_NDPI, help="Slide the mask is transferred onto (target).")
    parser.add_argument("--moving-annotations", default=None, help="GeoJSON export for the moving slide (default: qupath_deveral_slides_alignment/annotations/<stem>.geojson).")
    parser.add_argument("--fixed-annotations", default=None, help="GeoJSON export for the fixed slide (default: qupath_deveral_slides_alignment/annotations/<stem>.geojson).")
    parser.add_argument("--downsample", type=float, default=DEFAULT_DOWNSAMPLE, help="Target WSI pyramid downsample factor used for registration.")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated QuPath classification names to transfer/evaluate.")
    parser.add_argument("--methods", default="simpleitk,ants,voxelmorph", help="Comma-separated subset of: " + ", ".join(METHOD_FUNCS))
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    moving_annotations_path = args.moving_annotations or _default_annotation_path(args.moving_ndpi)
    fixed_annotations_path = args.fixed_annotations or _default_annotation_path(args.fixed_ndpi)
    for path in (moving_annotations_path, fixed_annotations_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Annotation export not found: {path}\n"
                "Export QuPath annotations to GeoJSON first by running "
                "qupath_deveral_slides_alignment/export_annotations.groovy "
                "('Run for project') inside QuPath."
            )

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_FUNCS:
            raise ValueError(f"Unknown method '{m}', choose from {list(METHOD_FUNCS)}")

    print(f"Loading moving slide (downsample~{args.downsample}): {args.moving_ndpi}")
    moving_rgb, moving_downsample = read_wsi_level(args.moving_ndpi, args.downsample)
    print(f"Loading fixed slide (downsample~{args.downsample}): {args.fixed_ndpi}")
    fixed_rgb, fixed_downsample = read_wsi_level(args.fixed_ndpi, args.downsample)

    moving_gray = rgb_to_gray(moving_rgb)
    fixed_gray = rgb_to_gray(fixed_rgb)

    print("Loading QuPath annotation masks...")
    moving_by_class = load_annotation_polygons(moving_annotations_path)
    fixed_by_class = load_annotation_polygons(fixed_annotations_path)

    moving_masks = {
        cls: rasterize_mask(moving_by_class.get(cls, []), moving_gray.shape, moving_downsample) for cls in classes
    }
    fixed_masks = {
        cls: rasterize_mask(fixed_by_class.get(cls, []), fixed_gray.shape, fixed_downsample) for cls in classes
    }

    print("Padding to a common canvas...")
    fixed_gray, moving_gray = pad_to_common_canvas([fixed_gray, moving_gray])
    height, width = fixed_gray.shape
    moving_masks = {cls: _pad_to_shape(m, height, width) for cls, m in moving_masks.items()}
    fixed_masks = {cls: _pad_to_shape(m, height, width) for cls, m in fixed_masks.items()}

    print("Running registration methods (this can take a while)...")
    results = [identity_result(moving_gray)]
    for m in methods:
        print(f"  -> {m}")
        results.append(METHOD_FUNCS[m](fixed_gray, moving_gray))

    overlay_dir = os.path.join(args.output_dir, "mask_transfer_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

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

    csv_path = os.path.join(args.output_dir, "mask_transfer_metrics.csv")
    df.to_csv(csv_path, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Mask transfer results ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved metrics table -> {csv_path}")
    print(f"Saved overlays -> {overlay_dir}")


if __name__ == "__main__":
    main()
