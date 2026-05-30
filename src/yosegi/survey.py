"""Detect the sample boundary in a low-magnification overview, plan a fine scan.

This module is the planning half of the *automatic whole-slide survey*: given
an overview image (e.g. a stitched coarse mosaic), find the bounding box of the
tissue and produce a list of stage positions that cover it with overlap at the
working magnification.

Segmentation is classical (Otsu intensity + local variance + morphological
close) so there is no ML dependency: tissue is darker and more textured than
the empty slide. Acquisition of the overview and execution of the planned scan
are *not* in this module yet -- they land when ``acquire`` learns an ``--auto``
flag in a follow-up. Errors are normalised into :class:`SurveyError` to match
``AcquisitionError`` / ``StitchError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PILImage

    ImageInput = str | Path | PILImage | np.ndarray


class SurveyError(RuntimeError):
    """Raised when a slide cannot be surveyed (no sample, bad inputs)."""


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
    components; any component whose area is below ``min_area_frac`` of the
    image is discarded as a speck. The returned :class:`BBox` is the union of
    all surviving components.

    ``image`` may be a path, a PIL image, or a NumPy array (HxW grayscale or
    HxWxC RGB). Raises :class:`SurveyError` if nothing above the area threshold
    is found, with guidance to check focus/exposure or pass a tighter ROI.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_closing, disk

    gray = _to_grayscale_u8(image)
    h, w = gray.shape
    if h == 0 or w == 0:
        raise SurveyError("overview image is empty")

    # Intensity cue: tissue is darker. Otsu fails on a flat image (single value),
    # so guard with a tiny dynamic-range check before calling it.
    intensity_mask = np.zeros_like(gray, dtype=bool)
    if int(gray.max()) - int(gray.min()) >= 5:
        try:
            t = threshold_otsu(gray)
            intensity_mask = gray < t
        except Exception:
            intensity_mask = np.zeros_like(gray, dtype=bool)

    # Texture cue: local variance. Anything above a small fraction of the
    # global variance counts as textured. Computed via two box means
    # (mean of squares minus square of mean) to avoid a per-pixel loop.
    fimg = gray.astype(np.float32)
    win = max(3, int(variance_window) | 1)  # force odd, >= 3
    local_mean = uniform_filter(fimg, size=win)
    local_var = uniform_filter(fimg * fimg, size=win) - local_mean * local_mean
    global_var = float(fimg.var())
    var_threshold = max(25.0, 0.1 * global_var)  # absolute floor handles flat backgrounds
    texture_mask = local_var > var_threshold

    mask = intensity_mask | texture_mask
    if close_radius > 0:
        mask = binary_closing(mask, disk(int(close_radius)))

    labelled = label(mask, connectivity=2)
    min_area = max(1, int(min_area_frac * h * w))
    components = [r for r in regionprops(labelled) if r.area >= min_area]
    if not components:
        raise SurveyError(
            "no sample detected in overview; check focus/exposure or pass a tighter ROI"
        )

    # Union of all kept components, in (y0, x0, y1, x1) skimage order -> BBox.
    y0 = min(r.bbox[0] for r in components)
    x0 = min(r.bbox[1] for r in components)
    y1 = max(r.bbox[2] for r in components)
    x1 = max(r.bbox[3] for r in components)
    return BBox(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1))


def _to_grayscale_u8(image: ImageInput) -> np.ndarray:
    """Coerce ``image`` to an ``uint8`` 2-D grayscale array.

    Accepts a path, PIL image, or NumPy array (grayscale or RGB / RGBA).
    Raises :class:`SurveyError` on unreadable paths or unsupported shapes.
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as im:
                return np.asarray(im.convert("L"), dtype=np.uint8)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise SurveyError(f"could not read overview image {image}: {exc}") from exc
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            # Drop alpha; use Rec. 601 luma weights for RGB -> gray.
            rgb = arr[..., :3].astype(np.float32)
            arr = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        elif arr.ndim != 2:
            raise SurveyError(f"unsupported image array shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
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

    # Project the bbox corners into stage space and take their axis-aligned span.
    # csm maps pixel deltas (dx, dy) -> stage deltas (dsx, dsy).
    corners_px = np.array(
        [[bbox.x0, bbox.y0], [bbox.x1, bbox.y0], [bbox.x1, bbox.y1], [bbox.x0, bbox.y1]],
        dtype=float,
    )
    corners_stage = corners_px @ csm.T  # apply affine to each corner
    sx_min, sy_min = corners_stage.min(axis=0)
    sx_max, sy_max = corners_stage.max(axis=0)
    span_x = sx_max - sx_min
    span_y = sy_max - sy_min

    # Stage extent of one tile = axis-aligned span of (tw, th) projected through csm.
    tile_corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=float) @ csm.T
    tile_span_x = float(tile_corners[:, 0].max() - tile_corners[:, 0].min())
    tile_span_y = float(tile_corners[:, 1].max() - tile_corners[:, 1].min())
    if tile_span_x <= 0 or tile_span_y <= 0:
        raise SurveyError("CSM projects the tile to zero stage extent -- check overview_csm")

    step_x = tile_span_x * (1.0 - overlap)
    step_y = tile_span_y * (1.0 - overlap)
    cols = max(1, int(np.ceil(span_x / step_x))) if span_x > tile_span_x else 1
    rows = max(1, int(np.ceil(span_y / step_y))) if span_y > tile_span_y else 1

    ox, oy = overview_origin_stage
    # Anchor the grid so the first tile's stage origin is the projected bbox corner.
    origin_x = ox + sx_min
    origin_y = oy + sy_min

    positions: list[tuple[int, int]] = []
    for r in range(rows):
        row_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in row_range:
            positions.append(
                (int(round(origin_x + c * step_x)), int(round(origin_y + r * step_y)))
            )

    bbox_stage = BBox(
        x0=int(round(ox + sx_min)),
        y0=int(round(oy + sy_min)),
        x1=int(round(ox + sx_max)),
        y1=int(round(oy + sy_max)),
    )
    return ScanPlan(
        positions=positions,
        rows=rows,
        cols=cols,
        step_x=int(round(step_x)),
        step_y=int(round(step_y)),
        bbox_stage=bbox_stage,
    )
