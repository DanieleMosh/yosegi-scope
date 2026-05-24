"""Lightweight data structures shared across acquisition and stitching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Tile:
    """A single captured image patch and where it sits in the scan grid.

    `row`/`col` are the tile's index in the raster grid (0-based). The optional
    `stage_x`/`stage_y`/`stage_z` record the OpenFlexure stage position (in steps)
    at capture time; these are also embedded in each tile's EXIF for the stitcher.
    """

    path: Path
    row: int
    col: int
    stage_x: int | None = None
    stage_y: int | None = None
    stage_z: int | None = None


@dataclass
class MosaicResult:
    """Summary of a completed stitch."""

    path: Path
    width: int
    height: int
    tile_count: int
