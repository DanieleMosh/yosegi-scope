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

    Stub -- implementation lands in the next commit.
    """
    raise NotImplementedError


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
