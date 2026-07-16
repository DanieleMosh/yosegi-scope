"""Detect the sample boundary in a low-magnification overview, plan a fine scan.

This module is the *automatic whole-slide survey*: given a way to drive the
scope, it captures a coarse overview pass, segments tissue vs empty slide on
the overview, plans a fine-magnification scan over the detected region, runs
it, and stitches the result.

Segmentation is classical (Otsu intensity + local variance + morphological
close) so there is no ML dependency: tissue is darker and more textured than
the empty slide. Errors are normalised into :class:`SurveyError` to match
``AcquisitionError`` / ``StitchError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from yosegi.acquire import AcquisitionError, Microscope, fetch_tiles, fetch_tiles_at_positions
from yosegi.models import MosaicResult
from yosegi.stitch import stitch_tiles

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PILImage

    ImageInput = str | Path | PILImage | np.ndarray


class SurveyError(RuntimeError):
    """Raised when a slide cannot be surveyed (no sample, bad inputs)."""


def _csm_from_manifest(manifest_path: Path) -> list[list[float]] | None:
    """Return the CSM matrix from an acquire manifest, or ``None`` if absent.

    Reads defensively: a missing file or unparseable JSON returns ``None`` rather
    than raising, so the caller can convert "no calibration" into a clean
    :class:`SurveyError` instead of leaking a ``FileNotFoundError`` /
    ``JSONDecodeError`` past the CLI's error normalisation.
    """
    import json

    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    csm = manifest.get("camera_stage_mapping")
    return csm if csm else None


@dataclass(frozen=True)
class BBox:
    """Pixel bounding box in an overview image.

    ``x0``/``y0`` are inclusive, ``x1``/``y1`` are exclusive (NumPy-style), so
    the box covers pixels ``[y0:y1, x0:x1]`` and has size ``(x1 - x0, y1 - y0)``.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0


@dataclass(frozen=True)
class ScanPlan:
    """A planned high-magnification scan over a detected sample region.

    ``positions`` are absolute stage ``(x, y)`` coordinates in **stage steps**,
    in snake (boustrophedon) order. ``rows``/``cols`` describe the grid;
    ``step_x``/``step_y`` is the stage motion between adjacent tiles (already
    accounts for the requested overlap and the camera-stage-mapping affine).
    ``bbox_stage`` is the detected bounding box transformed into stage steps so
    callers can log/render the planned region.
    """

    positions: list[tuple[int, int]]
    rows: int
    cols: int
    step_x: int
    step_y: int
    bbox_stage: BBox


def detect_sample_bbox(
    image: ImageInput,
    *,
    min_area_frac: float = 0.005,
    variance_window: int = 15,
    close_radius: int = 5,
) -> BBox:
    """Return the bounding box of the sample in an overview image.

    Combines two cues that distinguish tissue from an empty slide:
    Otsu-thresholded *intensity* (tissue is darker than the white background)
    and *local variance* (tissue is textured, the background is flat). The
    masks are OR-ed, morphologically closed, then split into connected
    components; components smaller than ``min_area_frac`` of the image are
    discarded as specks, and the bbox of the **largest remaining component**
    is returned. The largest-component policy keeps scattered noise (dust /
    vignetting / camera speckle near an empty corner) from being unioned with
    the real sample and inflating the bbox to the whole image.

    If the overview comes from a stitched canvas with un-tiled gaps (pixels at
    exactly RGB ``(0, 0, 0)``), those gaps are masked out before either cue is
    computed -- otherwise the canvas black would be misread as dark tissue and
    inflate the bbox to cover the whole canvas.

    ``image`` may be a path, a PIL image, or a NumPy array (HxW grayscale or
    HxWxC RGB). Raises :class:`SurveyError` if nothing above the area threshold
    is found, with guidance to check focus/exposure or pass a tighter ROI.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_closing, disk

    gray, canvas_mask = _to_grayscale_u8_with_canvas_mask(image)
    h, w = gray.shape
    if h == 0 or w == 0:
        raise SurveyError("overview image is empty")

    # Valid pixels = anything that isn't canvas-fill (RGB 0,0,0). For inputs
    # without canvas gaps, every pixel is valid.
    valid = ~canvas_mask
    if not valid.any():
        raise SurveyError("overview image has no valid (non-canvas) pixels")

    # Intensity cue: tissue is darker. Otsu fails on a flat image (single
    # value), so guard with a tiny dynamic-range check before calling it.
    # Compute the threshold over valid pixels only (canvas black would otherwise
    # pull Otsu's threshold and the dark mask down to nothing useful).
    intensity_mask = np.zeros_like(gray, dtype=bool)
    valid_pixels = gray[valid]
    if int(valid_pixels.max()) - int(valid_pixels.min()) >= 5:
        try:
            t = threshold_otsu(valid_pixels)
            intensity_mask = (gray < t) & valid
        except Exception:
            intensity_mask = np.zeros_like(gray, dtype=bool)

    # Texture cue: local variance. Anything above a small fraction of the
    # global variance counts as textured. Computed via two box means
    # (mean of squares minus square of mean) to avoid a per-pixel loop.
    fimg = gray.astype(np.float32)
    win = max(3, int(variance_window) | 1)  # force odd, >= 3
    local_mean = uniform_filter(fimg, size=win)
    local_var = uniform_filter(fimg * fimg, size=win) - local_mean * local_mean
    global_var = float(fimg[valid].var()) if valid.any() else 0.0
    var_threshold = max(25.0, 0.1 * global_var)  # absolute floor handles flat backgrounds
    texture_mask = (local_var > var_threshold) & valid

    mask = intensity_mask | texture_mask
    if close_radius > 0:
        mask = binary_closing(mask, disk(int(close_radius)))
    mask &= valid

    labelled = label(mask, connectivity=2)
    min_area = max(1, int(min_area_frac * h * w))
    components = [r for r in regionprops(labelled) if r.area >= min_area]
    if not components:
        raise SurveyError(
            "no sample detected in overview; check focus/exposure or pass a tighter ROI"
        )

    largest = max(components, key=lambda r: r.area)
    y0, x0, y1, x1 = largest.bbox
    return BBox(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1))


def _to_grayscale_u8_with_canvas_mask(image: ImageInput) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``image`` to an ``uint8`` grayscale array + canvas-fill mask.

    The canvas mask is ``True`` wherever the input had all RGB channels at
    exactly 0 -- the convention :func:`_stitch_overview_by_stage` and any
    stitcher use for "no tile placed here." For inputs that are already
    grayscale (or read from disk via ``Image.convert("L")``), the canvas mask
    is all-``False``: we cannot distinguish real-black from canvas-black once
    the colour information is gone, and a sample is never *all-zero* dark, so
    treating it as "no canvas gaps" is safe.

    Accepts a path, PIL image, or NumPy array (grayscale or RGB / RGBA).
    Raises :class:`SurveyError` on unreadable paths or unsupported shapes.
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    def _from_rgb_array(rgb_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        canvas = (rgb_arr[..., 0] == 0) & (rgb_arr[..., 1] == 0) & (rgb_arr[..., 2] == 0)
        f = rgb_arr.astype(np.float32)
        gray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        return np.clip(gray, 0, 255).astype(np.uint8), canvas

    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise SurveyError(f"could not read overview image {image}: {exc}") from exc
        return _from_rgb_array(rgb)
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return _from_rgb_array(rgb)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            if arr.shape[-1] >= 3:
                rgb = arr[..., :3]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                return _from_rgb_array(rgb)
            raise SurveyError(f"unsupported channel count {arr.shape[-1]}")
        if arr.ndim != 2:
            raise SurveyError(f"unsupported image array shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # No RGB available -> no way to distinguish canvas-black from real-dark.
        return arr, np.zeros(arr.shape, dtype=bool)
    raise SurveyError(f"unsupported image type {type(image).__name__}")


def plan_tile_grid(
    bbox: BBox,
    *,
    overview_origin_stage: tuple[int, int],
    overview_csm: list[list[float]],
    tile_size_px: tuple[int, int],
    overlap: float = 0.2,
) -> ScanPlan:
    """Plan a snake-ordered scan that covers ``bbox`` at the working magnification.

    The bounding box is given in *overview* pixels. ``overview_origin_stage`` is
    the absolute stage ``(x, y)`` (in steps) corresponding to overview pixel
    ``(0, 0)``, and ``overview_csm`` is the camera-stage-mapping matrix used
    for that overview (``stage_delta = csm @ pixel_delta``). For each axis the
    pixel-space bbox is projected into stage space, the per-tile stage motion
    is derived from ``tile_size_px`` and ``overlap``, and a snake-ordered grid
    of absolute stage positions is generated.

    Raises :class:`SurveyError` on invalid inputs (empty bbox, non-positive
    tile size, overlap not in ``[0, 1)``, or a degenerate CSM).
    """
    import numpy as np

    if bbox.is_empty:
        raise SurveyError("cannot plan a scan over an empty bounding box")
    tw, th = tile_size_px
    if tw <= 0 or th <= 0:
        raise SurveyError(f"tile_size_px must be positive, got {tile_size_px!r}")
    if not 0.0 <= overlap < 1.0:
        raise SurveyError(f"overlap must be in [0, 1), got {overlap!r}")
    csm = np.asarray(overview_csm, dtype=float)
    if csm.shape != (2, 2):
        raise SurveyError(f"overview_csm must be a 2x2 matrix, got shape {csm.shape}")

    # Plan the grid in image (pixel) space, where "overlap" is naturally
    # meaningful and the camera axes are aligned with the tile edges. Each tile
    # covers `tw x th` pixels in the bbox's coordinate frame; the per-tile
    # stride is `tile_size * (1 - overlap)` pixels. We then convert each tile's
    # origin from pixels to absolute stage coordinates via the CSM.
    #
    # This is critical when the CSM is rotated (e.g. ~90 degrees on
    # OpenFlexure): the image-x axis maps mostly to stage-y, so a single
    # camera-horizontal step requires moving the stage in y, not x. Computing
    # the stride in stage axes directly (as a previous version did) gives
    # tiles that are arranged along the wrong stage axis and never overlap.
    bbox_w = bbox.x1 - bbox.x0
    bbox_h = bbox.y1 - bbox.y0
    step_px_x = max(1.0, tw * (1.0 - overlap))
    step_px_y = max(1.0, th * (1.0 - overlap))
    cols = max(1, int(np.ceil(max(0, bbox_w - tw) / step_px_x)) + 1) if bbox_w > tw else 1
    rows = max(1, int(np.ceil(max(0, bbox_h - th) / step_px_y)) + 1) if bbox_h > th else 1

    # Convert per-step pixel motion to stage-step motion via the CSM. These
    # are 2-vectors -- one full vector per axis -- because a single image-axis
    # step in general moves the stage in BOTH x and y.
    col_step_stage = csm @ np.array([step_px_x, 0.0])
    row_step_stage = csm @ np.array([0.0, step_px_y])
    if np.allclose(col_step_stage, 0) or np.allclose(row_step_stage, 0):
        raise SurveyError("CSM projects a step to zero stage motion -- check overview_csm")

    ox, oy = overview_origin_stage
    # First tile's image-space origin is bbox.(x0, y0); stage origin is the
    # CSM-projected point relative to the overview's stage anchor.
    bbox_origin_stage = csm @ np.array([bbox.x0, bbox.y0])
    origin_stage_x = ox + bbox_origin_stage[0]
    origin_stage_y = oy + bbox_origin_stage[1]

    positions: list[tuple[int, int]] = []
    for r in range(rows):
        col_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in col_iter:
            stage = (
                origin_stage_x + c * col_step_stage[0] + r * row_step_stage[0],
                origin_stage_y + c * col_step_stage[1] + r * row_step_stage[1],
            )
            positions.append((int(round(stage[0])), int(round(stage[1]))))

    # bbox in stage coords: project all four pixel corners and take the axis-aligned span.
    corners_px = np.array(
        [[bbox.x0, bbox.y0], [bbox.x1, bbox.y0], [bbox.x1, bbox.y1], [bbox.x0, bbox.y1]],
        dtype=float,
    )
    corners_stage = corners_px @ csm.T
    sx_min, sy_min = corners_stage.min(axis=0)
    sx_max, sy_max = corners_stage.max(axis=0)
    bbox_stage = BBox(
        x0=int(round(ox + sx_min)),
        y0=int(round(oy + sy_min)),
        x1=int(round(ox + sx_max)),
        y1=int(round(oy + sy_max)),
    )

    # step_x / step_y are reported as the magnitude (Euclidean) of the per-axis
    # stage motion so callers logging "step" see a single sensible number even
    # when the move is diagonal in stage space.
    step_x = float(np.hypot(*col_step_stage))
    step_y = float(np.hypot(*row_step_stage))
    return ScanPlan(
        positions=positions,
        rows=rows,
        cols=cols,
        step_x=int(round(step_x)),
        step_y=int(round(step_y)),
        bbox_stage=bbox_stage,
    )


def run_auto_survey(
    client: Microscope,
    out_file: Path,
    *,
    overview_rows: int = 5,
    overview_cols: int = 5,
    overview_step_x: int = 2500,
    overview_step_y: int = 2500,
    overlap: float = 0.2,
    autofocus: bool = True,
    autofocus_once: bool = False,
    correlate: bool = True,
    high_pass_sigma: float = 10.0,
    minimum_overlap: float = 0.2,
    min_area_frac: float = 0.005,
) -> MosaicResult:
    """End-to-end automatic whole-slide survey, driven by an already-connected scope.

    Stages, in order:

    1. **Overview pass** -- coarse snake raster (``overview_rows`` x
       ``overview_cols``, ``overview_step_x``/``overview_step_y`` apart) into
       ``{out_file.stem}_overview/``. Same magnification as the high-res scan;
       only the step size is bigger.
    2. **Overview stitch** -- paste the overview tiles by stage + CSM into a
       single canvas (``{out_file.stem}_overview.jpg``). No correlation -- the
       overview only needs to be good enough for segmentation.
    3. **Detect** the sample bounding box on that canvas via
       :func:`detect_sample_bbox`.
    4. **Plan** a high-resolution scan over the bbox via :func:`plan_tile_grid`.
    5. **Execute** the plan with
       :func:`yosegi.acquire.fetch_tiles_at_positions` into
       ``{out_file.stem}_tiles/``.
    6. **Stitch** the high-res tiles to ``out_file`` via
       :func:`yosegi.stitch.stitch_tiles`.

    Raises :class:`AcquisitionError`, :class:`SurveyError`, or
    :class:`StitchError` depending on which stage fails -- the partial outputs
    on disk are left in place for debugging.
    """
    from PIL import Image

    out_file = Path(out_file)
    parent = out_file.parent
    stem = out_file.stem
    overview_dir = parent / f"{stem}_overview"
    overview_image_path = parent / f"{stem}_overview.jpg"
    tiles_dir = parent / f"{stem}_tiles"

    # Stage 1: overview raster (uses the scope's CSM, calibrating if needed).
    overview_tiles = fetch_tiles(
        host=None,
        out_dir=overview_dir,
        rows=overview_rows,
        cols=overview_cols,
        step_x=overview_step_x,
        step_y=overview_step_y,
        autofocus=autofocus,
        autofocus_once=autofocus_once,
        overlap=overlap,
        calibrate=True,
        client=client,
    )
    if not overview_tiles:
        raise AcquisitionError("overview pass produced no tiles")

    # Pull the CSM that ``acquire`` just embedded. Read it from the manifest
    # ``fetch_tiles`` wrote (same matrix the stitcher will see); guard the read
    # so a missing/corrupt manifest surfaces as a SurveyError, not a raw
    # traceback that escapes the CLI's error normalisation.
    csm = _csm_from_manifest(overview_dir / "manifest.json")
    if csm is None:
        raise SurveyError(
            "scope has no camera-stage-mapping calibration; run calibrate_xy() on the "
            "scope or use fetch_tiles with calibrate=True before surveying"
        )

    # Stage 2: stitch the overview by stage + CSM into one canvas. We do this
    # ourselves rather than via openflexure-stitching so we control the
    # pixel-to-stage origin exactly: pixel (0, 0) == (overview_origin_x_stage,
    # overview_origin_y_stage). That anchor is what plan_tile_grid needs.
    overview_image, overview_origin_stage = _stitch_overview_by_stage(overview_tiles, csm)
    overview_image.save(overview_image_path, "JPEG", quality=85)

    # Stage 3: detect bounding box on the overview.
    bbox = detect_sample_bbox(overview_image, min_area_frac=min_area_frac)

    # Stage 4: plan the high-res scan. tile_size_px is the working frame size,
    # which equals the overview tile size since we did not change objective.
    with Image.open(overview_tiles[0].path) as tile0:
        tile_w, tile_h = tile0.size
    plan = plan_tile_grid(
        bbox,
        overview_origin_stage=overview_origin_stage,
        overview_csm=csm,
        tile_size_px=(tile_w, tile_h),
        overlap=overlap,
    )

    # Stage 5: execute the plan.
    fetch_tiles_at_positions(
        client=client,
        out_dir=tiles_dir,
        positions=plan.positions,
        rows=plan.rows,
        cols=plan.cols,
        autofocus=autofocus,
        autofocus_once=autofocus_once,
    )

    # Stage 6: final stitch.
    return stitch_tiles(
        in_dir=tiles_dir,
        out_file=out_file,
        correlate=correlate,
        high_pass_sigma=high_pass_sigma,
        minimum_overlap=minimum_overlap,
    )


def _stitch_overview_by_stage(
    overview_tiles: list,
    csm: list[list[float]],
) -> tuple[PILImage, tuple[int, int]]:
    """Paste overview tiles onto a single canvas using stage + CSM only.

    Returns ``(canvas, origin_stage)`` where ``origin_stage`` is the absolute
    stage ``(x, y)`` (in steps) that corresponds to pixel ``(0, 0)`` of the
    canvas. With ``stage_delta = csm @ pixel_delta``, the canvas pixel for a
    tile at ``stage_i`` is ``csm^{-1} @ (stage_i - stage_min)``; we offset by
    the minimum projected pixel so all positions are non-negative.
    """
    import numpy as np
    from PIL import Image

    if not overview_tiles:
        raise SurveyError("overview tile list is empty")

    csm_arr = np.asarray(csm, dtype=float)
    if csm_arr.shape != (2, 2):
        raise SurveyError(f"overview CSM must be 2x2, got shape {csm_arr.shape}")
    try:
        csm_inv = np.linalg.inv(csm_arr)
    except np.linalg.LinAlgError as exc:
        raise SurveyError(f"overview CSM is singular: {exc}") from exc

    stages = np.array([[t.stage_x, t.stage_y] for t in overview_tiles], dtype=float)
    stage_min = stages.min(axis=0)
    deltas_stage = stages - stage_min
    pixel_positions = deltas_stage @ csm_inv.T  # (N, 2) -- (px, py)

    with Image.open(overview_tiles[0].path) as t0:
        tile_w, tile_h = t0.size

    # Pad by the most-negative projection so all tiles land at non-negative
    # pixel positions. ``csm`` may rotate/flip so projected pixels can go negative.
    canvas_min = pixel_positions.min(axis=0)
    shifted = pixel_positions - canvas_min  # now all >= 0
    canvas_max = shifted.max(axis=0)
    canvas_w = int(np.ceil(canvas_max[0] + tile_w))
    canvas_h = int(np.ceil(canvas_max[1] + tile_h))

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    for tile, (px, py) in zip(overview_tiles, shifted, strict=True):
        with Image.open(tile.path) as im:
            canvas.paste(im.convert("RGB"), (int(round(px)), int(round(py))))

    # Origin stage = stage at pixel (0, 0). Pixel (0, 0) corresponds to
    # stage_min + csm @ canvas_min (since we subtracted canvas_min above).
    origin_stage_xy = stage_min + csm_arr @ canvas_min
    origin_stage = (int(round(origin_stage_xy[0])), int(round(origin_stage_xy[1])))
    return canvas, origin_stage


__all__ = [
    "BBox",
    "ScanPlan",
    "SurveyError",
    "detect_sample_bbox",
    "plan_tile_grid",
    "run_auto_survey",
]
