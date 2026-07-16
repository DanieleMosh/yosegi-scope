"""Detect tissue regions in a low-magnification overview, plan a fine scan.

This module is the *automatic whole-slide survey*: given a way to drive the
scope, it captures a coarse overview pass, segments tissue vs empty slide on
the overview, plans a fine-magnification scan **only over the tissue** (skipping
empty area), runs it, and stitches the result.

Segmentation is classical (no ML dependency): tissue is separated from the empty
slide by three complementary cues combined into one boolean mask --

* **saturation** -- stained tissue is coloured while glass is near-grey, so a
  high saturation channel is the cleanest tissue cue (the CLAM / digital-
  pathology approach). Weak on faint / unstained brightfield samples, so it is
  backed by:
* **intensity** -- tissue is darker than the bright background (Otsu), and
* **texture** -- tissue is textured while the background is flat (local variance).

The mask is split into connected components (:func:`detect_sample_regions`);
each surviving component is one tissue :class:`Region`. :func:`plan_survey`
rasters a tile grid over the regions and keeps only tiles whose centre falls on
tissue, so multiple sections on one slide are all scanned and blank tiles are
skipped. :func:`detect_sample_bbox` is retained as a thin single-bbox wrapper.

Errors are normalised into :class:`SurveyError` to match ``AcquisitionError`` /
``StitchError``.
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

    from yosegi.models import Tile

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
class Region:
    """One detected tissue region in an overview image.

    ``bbox`` is its pixel bounding box; ``area`` is the component's pixel count;
    ``centroid`` is ``(y, x)`` in pixels. The full boolean footprint lives in the
    parent :class:`TissueMask` (this is a lightweight per-region descriptor).
    """

    bbox: BBox
    area: int
    centroid: tuple[float, float]


@dataclass(frozen=True)
class TissueMask:
    """The result of tissue detection on an overview.

    ``mask`` is an ``HxW`` boolean array (``True`` = tissue) used to gate tiles;
    ``regions`` are the surviving connected components, largest first. ``bbox`` is
    the union pixel bbox of all regions (the extent to raster).
    """

    mask: np.ndarray
    regions: list[Region]
    bbox: BBox

    @property
    def shape(self) -> tuple[int, int]:
        return (self.mask.shape[0], self.mask.shape[1])


@dataclass(frozen=True)
class ScanPlan:
    """A planned high-magnification scan over the detected tissue.

    ``positions`` are absolute stage ``(x, y)`` coordinates in **stage steps**,
    in snake (boustrophedon) order, one per tile actually kept. ``rowcols`` gives
    the ``(row, col)`` grid index of each position (parallel to ``positions``),
    densely renumbered over the raster; a tissue-gated plan is **sparse**, so
    ``len(positions)`` may be less than ``rows * cols``. ``rows``/``cols`` are the
    full raster extent; ``step_x``/``step_y`` is the stage motion between adjacent
    grid cells (accounts for overlap and the CSM affine). ``bbox_stage`` is the
    detected region transformed into stage steps for logging/rendering.
    """

    positions: list[tuple[int, int]]
    rowcols: list[tuple[int, int]]
    rows: int
    cols: int
    step_x: int
    step_y: int
    bbox_stage: BBox

    @property
    def tile_count(self) -> int:
        return len(self.positions)


def _tissue_mask_array(
    image: ImageInput,
    *,
    variance_window: int,
    close_radius: int,
) -> np.ndarray:
    """Compute the raw (pre-component-filter) boolean tissue mask for ``image``.

    Combines three complementary cues over the valid (non-canvas) pixels and
    OR-s them: **saturation** (stained tissue is coloured, glass is near-grey),
    **intensity** (tissue is darker than the bright background, via Otsu), and
    **texture** (tissue is textured, background is flat, via local variance).
    Saturation is the cleanest cue but useless on faint/unstained brightfield,
    where intensity + texture carry it -- so all three are unioned rather than
    relied on individually. The union is morphologically closed and clipped back
    to the valid region. Canvas-fill gaps (exact RGB ``(0,0,0)``) are excluded so
    they are never misread as dark tissue.

    Returns an ``HxW`` boolean array. Raises :class:`SurveyError` on an empty or
    all-canvas image.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk

    rgb, gray, canvas_mask = _to_arrays_with_canvas_mask(image)
    h, w = gray.shape
    if h == 0 or w == 0:
        raise SurveyError("overview image is empty")

    valid = ~canvas_mask
    if not valid.any():
        raise SurveyError("overview image has no valid (non-canvas) pixels")

    def _otsu_dark(channel: np.ndarray, *, invert: bool) -> np.ndarray:
        """Otsu-threshold ``channel`` over valid pixels; guard flat inputs."""
        vals = channel[valid]
        if int(vals.max()) - int(vals.min()) < 5:
            return np.zeros_like(gray, dtype=bool)
        try:
            t = threshold_otsu(vals)
        except ValueError:
            # threshold_otsu raises ValueError on a degenerate (near-constant)
            # histogram the range check above didn't already catch.
            return np.zeros_like(gray, dtype=bool)
        picked = (channel > t) if invert else (channel < t)
        return picked & valid

    # Saturation cue: high S = coloured (stained) tissue. Only meaningful when
    # the overview is colour (RGB); grayscale inputs give S == 0 everywhere.
    if rgb is not None:
        from skimage.color import rgb2hsv

        sat = (rgb2hsv(rgb)[..., 1] * 255.0).astype(np.uint8)
        saturation_mask = _otsu_dark(sat, invert=True)  # keep high-saturation pixels
    else:
        saturation_mask = np.zeros_like(gray, dtype=bool)

    # Intensity cue: tissue is darker than the bright background.
    intensity_mask = _otsu_dark(gray, invert=False)

    # Texture cue: local variance (mean of squares minus square of mean),
    # vectorised via two box filters. Above an absolute floor OR a fraction of
    # the global variance counts as textured.
    fimg = gray.astype(np.float32)
    win = max(3, int(variance_window) | 1)  # force odd, >= 3
    local_mean = uniform_filter(fimg, size=win)
    local_var = uniform_filter(fimg * fimg, size=win) - local_mean * local_mean
    global_var = float(fimg[valid].var())
    var_threshold = max(25.0, 0.1 * global_var)
    texture_mask = (local_var > var_threshold) & valid

    mask = saturation_mask | intensity_mask | texture_mask
    if close_radius > 0:
        mask = binary_closing(mask, disk(int(close_radius)))
    # Closing can grow the mask into the canvas gutter; clip it back so no
    # detected component ever includes a canvas-fill pixel.
    mask &= valid
    return mask


def detect_sample_regions(
    image: ImageInput,
    *,
    min_area_frac: float = 0.005,
    variance_window: int = 15,
    close_radius: int = 5,
    max_regions: int | None = None,
) -> TissueMask:
    """Detect every tissue region in an overview image.

    Builds the tissue mask (:func:`_tissue_mask_array`), splits it into connected
    components, and keeps each component whose area is at least ``min_area_frac``
    of the image as one :class:`Region` (largest first). Unlike a single-bbox
    detector this returns *all* qualifying regions, so a slide carrying several
    sections is fully surveyed. ``max_regions`` caps the number kept (largest by
    area) when set.

    ``image`` may be a path, a PIL image, or a NumPy array (HxW grayscale or
    HxWxC RGB). Raises :class:`SurveyError` if no region clears the area
    threshold, with guidance to check focus/exposure or pass a tighter ROI.
    """
    import numpy as np
    from skimage.measure import label, regionprops

    mask = _tissue_mask_array(image, variance_window=variance_window, close_radius=close_radius)
    h, w = mask.shape

    labelled = label(mask, connectivity=2)
    min_area = max(1, int(min_area_frac * h * w))
    props = [r for r in regionprops(labelled) if r.area >= min_area]
    if not props:
        raise SurveyError(
            "no sample detected in overview; check focus/exposure or pass a tighter ROI"
        )
    props.sort(key=lambda r: r.area, reverse=True)
    if max_regions is not None and max_regions > 0:
        props = props[:max_regions]

    # Rebuild the mask from only the kept components so plan_survey's tile gate
    # never lets a below-threshold speck admit a tile.
    kept_labels = {r.label for r in props}
    kept_mask = np.isin(labelled, list(kept_labels))

    regions: list[Region] = []
    for r in props:
        y0, x0, y1, x1 = r.bbox
        cy, cx = r.centroid
        regions.append(
            Region(bbox=BBox(int(x0), int(y0), int(x1), int(y1)), area=int(r.area),
                   centroid=(float(cy), float(cx)))
        )

    x0 = min(reg.bbox.x0 for reg in regions)
    y0 = min(reg.bbox.y0 for reg in regions)
    x1 = max(reg.bbox.x1 for reg in regions)
    y1 = max(reg.bbox.y1 for reg in regions)
    return TissueMask(mask=kept_mask, regions=regions, bbox=BBox(x0, y0, x1, y1))


def detect_sample_bbox(
    image: ImageInput,
    *,
    min_area_frac: float = 0.005,
    variance_window: int = 15,
    close_radius: int = 5,
) -> BBox:
    """Return the bounding box of the largest tissue region in an overview.

    Thin wrapper over :func:`detect_sample_regions` kept for backward
    compatibility: it returns the bbox of the single largest region. Prefer
    :func:`detect_sample_regions` for multi-region slides and
    :func:`plan_survey` for a tissue-gated scan that skips empty tiles.
    """
    tissue = detect_sample_regions(
        image, min_area_frac=min_area_frac, variance_window=variance_window,
        close_radius=close_radius,
    )
    return tissue.regions[0].bbox


def _to_arrays_with_canvas_mask(
    image: ImageInput,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Coerce ``image`` to ``(rgb_u8_or_None, gray_u8, canvas_mask)``.

    ``rgb`` is the ``HxWx3`` uint8 array when colour is available (needed for the
    saturation cue), else ``None`` for grayscale-only inputs. ``gray`` is the
    ``uint8`` luminance. ``canvas_mask`` is ``True`` wherever the input had all
    RGB channels at exactly 0 -- the "no tile placed here" convention used by
    :func:`_stitch_overview_by_stage`. Grayscale inputs get an all-``False``
    canvas mask: real-black and canvas-black are indistinguishable once colour is
    gone, and a sample is never *all-zero* dark, so "no gaps" is the safe read.

    Accepts a path, PIL image, or NumPy array (grayscale or RGB / RGBA).
    Raises :class:`SurveyError` on unreadable paths or unsupported shapes.
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    def _from_rgb_array(rgb_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        canvas = (rgb_arr[..., 0] == 0) & (rgb_arr[..., 1] == 0) & (rgb_arr[..., 2] == 0)
        f = rgb_arr.astype(np.float32)
        gray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        return rgb_arr, np.clip(gray, 0, 255).astype(np.uint8), canvas

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
        # No RGB available -> no saturation cue, and no way to distinguish
        # canvas-black from real-dark.
        return None, arr, np.zeros(arr.shape, dtype=bool)
    raise SurveyError(f"unsupported image type {type(image).__name__}")


class _GridGeometry:
    """Shared image-space raster geometry over a pixel ``bbox``.

    Encapsulates the pixel-to-stage conversion so both :func:`plan_tile_grid`
    (dense) and :func:`plan_survey` (tissue-gated, sparse) compute identical tile
    positions and only differ in which cells they keep.
    """

    def __init__(
        self,
        bbox: BBox,
        *,
        overview_origin_stage: tuple[int, int],
        overview_csm: list[list[float]],
        tile_size_px: tuple[int, int],
        overlap: float,
    ) -> None:
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
        # meaningful and the camera axes align with the tile edges; convert each
        # cell's origin to stage steps via the CSM. This matters when the CSM is
        # rotated (~90 deg on OpenFlexure): image-x maps mostly to stage-y, so a
        # camera-horizontal step moves the stage in y. Stepping along stage axes
        # directly gives tiles arranged along the wrong axis that never overlap.
        bbox_w = bbox.x1 - bbox.x0
        bbox_h = bbox.y1 - bbox.y0
        self.tw, self.th = tw, th
        self.step_px_x = max(1.0, tw * (1.0 - overlap))
        self.step_px_y = max(1.0, th * (1.0 - overlap))
        self.cols = (
            max(1, int(np.ceil(max(0, bbox_w - tw) / self.step_px_x)) + 1) if bbox_w > tw else 1
        )
        self.rows = (
            max(1, int(np.ceil(max(0, bbox_h - th) / self.step_px_y)) + 1) if bbox_h > th else 1
        )

        # Per-step stage motion is a 2-vector per axis: one image-axis step moves
        # the stage in both x and y in general.
        self.col_step_stage = csm @ np.array([self.step_px_x, 0.0])
        self.row_step_stage = csm @ np.array([0.0, self.step_px_y])
        if np.allclose(self.col_step_stage, 0) or np.allclose(self.row_step_stage, 0):
            raise SurveyError("CSM projects a step to zero stage motion -- check overview_csm")

        self.bbox = bbox
        self.csm = csm
        self.ox, self.oy = overview_origin_stage
        bbox_origin_stage = csm @ np.array([bbox.x0, bbox.y0])
        self.origin_stage_x = self.ox + bbox_origin_stage[0]
        self.origin_stage_y = self.oy + bbox_origin_stage[1]

    def tile_center_px(self, r: int, c: int) -> tuple[float, float]:
        """Centre of grid cell ``(r, c)`` in *overview-image* pixel coords."""
        return (
            self.bbox.x0 + c * self.step_px_x + self.tw / 2.0,
            self.bbox.y0 + r * self.step_px_y + self.th / 2.0,
        )

    def tile_stage(self, r: int, c: int) -> tuple[int, int]:
        """Absolute stage ``(x, y)`` for the origin of grid cell ``(r, c)``."""
        sx = self.origin_stage_x + c * self.col_step_stage[0] + r * self.row_step_stage[0]
        sy = self.origin_stage_y + c * self.col_step_stage[1] + r * self.row_step_stage[1]
        return (int(round(sx)), int(round(sy)))

    def bbox_stage(self) -> BBox:
        """The pixel ``bbox`` projected into an axis-aligned stage-step box."""
        import numpy as np

        b = self.bbox
        corners_px = np.array(
            [[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], dtype=float
        )
        corners_stage = corners_px @ self.csm.T
        sx_min, sy_min = corners_stage.min(axis=0)
        sx_max, sy_max = corners_stage.max(axis=0)
        return BBox(
            x0=int(round(self.ox + sx_min)),
            y0=int(round(self.oy + sy_min)),
            x1=int(round(self.ox + sx_max)),
            y1=int(round(self.oy + sy_max)),
        )

    def step_magnitudes(self) -> tuple[int, int]:
        """Euclidean per-axis stage step, for callers that log a single number."""
        import numpy as np

        return (
            int(round(float(np.hypot(*self.col_step_stage)))),
            int(round(float(np.hypot(*self.row_step_stage)))),
        )


def plan_tile_grid(
    bbox: BBox,
    *,
    overview_origin_stage: tuple[int, int],
    overview_csm: list[list[float]],
    tile_size_px: tuple[int, int],
    overlap: float = 0.2,
) -> ScanPlan:
    """Plan a **dense** snake-ordered scan covering ``bbox`` at working magnification.

    Every grid cell over the pixel ``bbox`` is kept (no tissue gate), so
    ``len(positions) == rows * cols``. ``overview_origin_stage`` is the absolute
    stage ``(x, y)`` at overview pixel ``(0, 0)``; ``overview_csm`` maps pixel
    deltas to stage deltas. For a tissue-gated scan that skips blank tiles use
    :func:`plan_survey`.

    Raises :class:`SurveyError` on invalid inputs (empty bbox, non-positive tile
    size, overlap not in ``[0, 1)``, or a degenerate CSM).
    """
    geo = _GridGeometry(
        bbox,
        overview_origin_stage=overview_origin_stage,
        overview_csm=overview_csm,
        tile_size_px=tile_size_px,
        overlap=overlap,
    )
    positions: list[tuple[int, int]] = []
    rowcols: list[tuple[int, int]] = []
    for r in range(geo.rows):
        col_iter = range(geo.cols) if r % 2 == 0 else range(geo.cols - 1, -1, -1)
        for c in col_iter:
            positions.append(geo.tile_stage(r, c))
            rowcols.append((r, c))
    step_x, step_y = geo.step_magnitudes()
    return ScanPlan(
        positions=positions,
        rowcols=rowcols,
        rows=geo.rows,
        cols=geo.cols,
        step_x=step_x,
        step_y=step_y,
        bbox_stage=geo.bbox_stage(),
    )


def plan_survey(
    tissue: TissueMask,
    *,
    overview_origin_stage: tuple[int, int],
    overview_csm: list[list[float]],
    tile_size_px: tuple[int, int],
    overlap: float = 0.2,
) -> ScanPlan:
    """Plan a **tissue-gated** snake scan over detected tissue, skipping blank tiles.

    Rasters the same snake grid as :func:`plan_tile_grid` over the union bbox of
    ``tissue.regions``, but keeps a cell only if its centre pixel lands on tissue
    (``tissue.mask``). Blank cells between and around regions are dropped, so a
    slide with several sections is covered without scanning the empty slide in
    between. The kept cells are renumbered densely into snake-ordered
    ``(row, col)`` (``rowcols``) so tile filenames stay well-formed.

    Falls back to keeping every cell in the union bbox if the gate would leave no
    tiles (e.g. tiles larger than the tissue speck), so a scan is never empty.

    Raises :class:`SurveyError` on the same invalid inputs as
    :func:`plan_tile_grid`, or if ``tissue`` has no regions.
    """
    if not tissue.regions:
        raise SurveyError("cannot plan a survey with no detected tissue regions")
    geo = _GridGeometry(
        tissue.bbox,
        overview_origin_stage=overview_origin_stage,
        overview_csm=overview_csm,
        tile_size_px=tile_size_px,
        overlap=overlap,
    )
    mask = tissue.mask
    mh, mw = mask.shape

    def keep(r: int, c: int) -> bool:
        cx, cy = geo.tile_center_px(r, c)
        icx, icy = int(round(cx)), int(round(cy))
        if 0 <= icy < mh and 0 <= icx < mw:
            return bool(mask[icy, icx])
        return False

    positions: list[tuple[int, int]] = []
    rowcols: list[tuple[int, int]] = []
    for r in range(geo.rows):
        col_iter = range(geo.cols) if r % 2 == 0 else range(geo.cols - 1, -1, -1)
        for c in col_iter:
            if keep(r, c):
                positions.append(geo.tile_stage(r, c))
                rowcols.append((r, c))

    if not positions:
        # Every tile centre missed the tissue (tissue smaller than one tile).
        # Scan the whole union bbox rather than returning an empty plan.
        for r in range(geo.rows):
            col_iter = range(geo.cols) if r % 2 == 0 else range(geo.cols - 1, -1, -1)
            for c in col_iter:
                positions.append(geo.tile_stage(r, c))
                rowcols.append((r, c))

    # Renumber kept cells densely so filenames are compact and gap-free while
    # preserving the snake visit order.
    dense_rowcols = _densify_rowcols(rowcols)
    step_x, step_y = geo.step_magnitudes()
    return ScanPlan(
        positions=positions,
        rowcols=dense_rowcols,
        rows=geo.rows,
        cols=geo.cols,
        step_x=step_x,
        step_y=step_y,
        bbox_stage=geo.bbox_stage(),
    )


def _densify_rowcols(rowcols: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Renumber sparse ``(row, col)`` to gap-free indices, preserving order.

    Rows keep their relative order (0, 1, 2, ...) and within each kept row the
    columns are renumbered 0..k in visit order, so filenames like
    ``tile_r00_c00`` stay compact even though the original grid was sparse.
    """
    row_order: list[int] = []
    for r, _ in rowcols:
        if r not in row_order:
            row_order.append(r)
    row_index = {r: i for i, r in enumerate(row_order)}

    dense: list[tuple[int, int]] = []
    seen_in_row: dict[int, int] = {}
    for r, _ in rowcols:
        dr = row_index[r]
        dc = seen_in_row.get(dr, 0)
        seen_in_row[dr] = dc + 1
        dense.append((dr, dc))
    return dense


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
    max_regions: int | None = None,
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
    3. **Detect** every tissue region on that canvas via
       :func:`detect_sample_regions` (``max_regions`` caps how many are kept).
    4. **Plan** a tissue-gated high-resolution scan over the regions via
       :func:`plan_survey` -- tiles whose centre misses the tissue are skipped,
       so blank slide between sections is not scanned.
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
    # overview_origin_y_stage). That anchor is what plan_survey needs. Detection
    # runs on the in-memory canvas (not the re-read JPEG) so JPEG compression
    # can't perturb the exact (0,0,0) canvas-gap convention.
    overview_image, overview_origin_stage = _stitch_overview_by_stage(overview_tiles, csm)
    overview_image.save(overview_image_path, "JPEG", quality=85)

    # Stage 3: detect every tissue region on the overview.
    tissue = detect_sample_regions(
        overview_image, min_area_frac=min_area_frac, max_regions=max_regions
    )

    # Stage 4: plan the tissue-gated high-res scan. tile_size_px is the working
    # frame size, which equals the overview tile size since we did not change
    # objective.
    with Image.open(overview_tiles[0].path) as tile0:
        tile_w, tile_h = tile0.size
    plan = plan_survey(
        tissue,
        overview_origin_stage=overview_origin_stage,
        overview_csm=csm,
        tile_size_px=(tile_w, tile_h),
        overlap=overlap,
    )

    # Stage 5: execute the (possibly sparse) plan, passing the densely-renumbered
    # (row, col) so tile filenames stay compact.
    fetch_tiles_at_positions(
        client=client,
        out_dir=tiles_dir,
        positions=plan.positions,
        rows=plan.rows,
        cols=plan.cols,
        rowcols=plan.rowcols,
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
    overview_tiles: list[Tile],
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
    "Region",
    "ScanPlan",
    "SurveyError",
    "TissueMask",
    "detect_sample_bbox",
    "detect_sample_regions",
    "plan_survey",
    "plan_tile_grid",
    "run_auto_survey",
]
