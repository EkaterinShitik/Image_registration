# Merge image

Benchmarking and mask-transfer toolkit for non-rigid registration of
whole-slide histology images, built on three independent backends:
**SimpleITK** (B-spline FFD), **ANTsPy** (SyN diffeomorphic) and
**VoxelMorph** (instance-optimized CNN).

Three entry points cover three scenarios:

- `main.py` - generic benchmark on a pair of pre-aligned sample JPEGs
  (`data/108.jpeg` -> `data/109.jpeg`), scoring all three backends.
- `main_mask.py` - transfers a QuPath pixel-classifier mask (e.g.
  Tissue/Blood) between **two adjacent serial H&E sections** of the same
  tissue block.
- `main_mask_comet.py` - transfers QuPath Tissue/Nucleus masks between
  **two scans of the same physical slide** acquired with different
  instruments/modalities: H&E brightfield and COMET multiplex fluorescence.

## 1. The core idea: registration and mask transfer

Two images of (approximately) the same tissue are never pixel-aligned out of
the box - the slide is remounted, re-scanned, or imaged with a different
instrument, so the tissue appears shifted, rotated, scaled and/or locally
warped between the two scans. **Image registration** estimates the spatial
transformation that maps one image (the **moving** image) onto the other's
coordinate grid (the **fixed** image).

All three backends here produce a dense, non-rigid **displacement field**
`field[row, col] = (d_row, d_col)`: for every pixel of the fixed image, the
field gives the offset to the corresponding pixel in the moving image. This
is a superset of a simple rigid/affine transform - it also captures local
tissue deformation (stretching, tearing, non-uniform shrinkage) that a single
global transform cannot.

Once the field is known, the same transform used to warp the moving
*image* onto the fixed grid is reused to warp the moving *mask*
(`warp_mask`, nearest-neighbor interpolation to keep it binary) and to map
individual *points* (`transform_points`, used for landmark-based metrics).
This is the actual goal of the pipeline: a pixel-classifier mask (Tissue,
Nucleus, Blood, ...) computed by QuPath on one slide is transferred onto a
second slide where no such classification exists yet, and then checked
against that second slide's own independently-computed mask to quantify how
trustworthy the transferred mask is.

## 2. Metrics - what to look at

Each method is scored with a mix of intensity-based, geometry-based and
deformation-quality metrics (see `metrics.py`). No single metric tells the
whole story, so look at them together:

- **Dice (DSC) / IoU / Precision / Recall** - overlap between the warped
  mask and the real mask on the target image. Dice/IoU summarize overall
  overlap; Precision and Recall tell you *which way* a method is biased
  (e.g. high Recall + low Precision means the transferred mask is too
  large/leaky, not too small).
- **HD95 (95th-percentile Hausdorff Distance)** - worst-case boundary
  error, in pixels, ignoring the top 5% of outliers. Dice can look good even
  when a mask's boundary is locally far off in one region (e.g. a torn
  tissue fold); HD95 catches that.
- **NMI (Normalized Mutual Information)** - how well the warped
  *grayscale image* intensities statistically match the fixed image,
  independent of any mask. Useful as a registration-quality sanity check
  even without an evaluation mask, and robust to cross-modality intensity
  differences (unlike raw correlation).
- **TRE (Target Registration Error)** - mean Euclidean distance (px)
  between independent landmark correspondences (ORB + RANSAC, see
  `landmarks.py`) after applying `transform_points`. This is the only metric
  that never looks at the mask at all, so it is a good cross-check that the
  mask metrics above aren't misleadingly good/bad due to the mask itself.
- **Jacobian determinant of the displacement field (`jac_negative_pct`)**
  - the percentage of pixels where the deformation folds onto itself
  (non-invertible, anatomically implausible). A method can score well on
  Dice/HD95 while quietly producing a folded, physically nonsensical
  deformation elsewhere - this metric is what catches that.
- **Runtime** - SimpleITK and ANTs are CPU, multi-resolution, and
  comparatively fast; VoxelMorph is instance-optimized per image pair (no
  pretrained model for this modality) and is typically the slowest of the
  three.

In short: use Dice/IoU/HD95 to judge the transferred mask itself, NMI/TRE as
mask-independent sanity checks on the underlying registration, and the
Jacobian stats to make sure a good Dice score isn't hiding a degenerate
deformation.

## 3. HE ↔ HE vs. HE ↔ COMET registration

`main_mask.py` and `main_mask_comet.py` share the same three registration
backends and the same evaluation metrics, but they start from very different
inputs, which is why `main_mask_comet.py` adds a whole pre-processing stage
that `main_mask.py` doesn't need:

| | `main_mask.py` (HE ↔ HE) | `main_mask_comet.py` (HE ↔ COMET) |
|---|---|---|
| Slides | Two **different**, adjacent serial sections | **One** physical slide, scanned twice |
| Modality | Same (H&E brightfield) on both | **Different**: H&E brightfield vs. COMET multiplex fluorescence |
| Resolution | Both read at the same pyramid downsample factor | Each slide resampled to the same physical **microns-per-pixel**, since the two scanners' pyramids don't share a common downsample |
| Channels | RGB -> grayscale directly | COMET is multichannel UINT16; one channel (default: DAPI) is selected, percentile-stretched and inverted to brightfield polarity |
| Gross misalignment | Negligible (consecutive sections, same scanner) | Can include a large translation and even a 90-degree rotation between the two scans |
| Pre-processing | Pad to a common canvas only | Crop to Tissue bbox, 90-degree **coarse pre-alignment**, histogram matching, then pad to a common canvas |

Methods/steps used **only** for HE ↔ COMET (`main_mask_comet.py`):

- **Physical-resolution matching** (`read_he_level`/`read_comet_level`,
  `DEFAULT_TARGET_MPP`) - each slide is resampled (bilinear) to an exact
  target microns/pixel value instead of relying on native pyramid levels,
  since the two scanners' downsample factors don't line up.
- **COMET channel selection + polarity inversion** - the DAPI channel is
  percentile-stretched to `[0, 1]` and inverted so nuclei are dark on a
  bright background, matching H&E's brightfield polarity that the
  registration backends were tuned against; the background outside Tissue is
  then forced to white to mimic H&E's blank slide background.
- **Crop to Tissue bounding box** (`crop_to_mask_bbox`) - anchors each
  slide's tissue near the top-left corner before padding, since the two
  scanners' canvas/stage extents can differ substantially.
- **Coarse 90-degree pre-alignment** (`coarse_prealign`) - searches the 4
  axis-aligned rotations of the moving image and, for each, estimates the
  residual translation via FFT phase correlation, keeping the
  rotation/shift with the lowest residual error. This runs *before* the fine
  non-rigid registration because the backends' own affine/rigid stages only
  have a narrow capture range and cannot recover from a 90-degree rotation or
  a large shift on their own.
- **Histogram matching** (`match_histogram_to_reference`) - remaps the H&E
  grayscale distribution to COMET's, since intensity-based similarity
  metrics (mutual information, local NCC) assume the two images relate
  through a roughly comparable brightness/contrast distribution, which
  cross-modality pairs otherwise lack even after inversion.
- **ANTs staging choice** - `main_mask_comet.py` exposes `--ants-transform
  SyNRA` (Rigid -> Affine -> SyN) vs. `SyN` (Affine -> SyN only), since the
  wider capture range of an explicit rigid stage matters more for a harder,
  cross-modality pair than for same-modality serial sections.

Everything downstream of these extra steps - padding to a common canvas,
running SimpleITK/ANTs/VoxelMorph, warping masks, and scoring with
Dice/IoU/Precision/Recall/HD95 - is identical between the two scripts.
