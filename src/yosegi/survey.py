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

    Stub -- implementation lands in the next commit.
    """
    raise NotImplementedError
