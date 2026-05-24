"""Align overlapping tiles and merge them into one seamless composite.

This is a documented stub. The real implementation will use ``m2stitch`` (a
MIST-inspired grid stitcher) plus Pillow for compositing::

    import m2stitch

    # images: (N, H, W) array of grayscale tiles, with parallel row/col index lists
    result_df, _ = m2stitch.stitch_images(images, rows, cols)
    # result_df has refined per-tile "x_pos"/"y_pos" pixel offsets

Sketch of the steps to fill in here:

1. Discover tiles in ``in_dir`` and recover each tile's (row, col) grid index
   (from filename, a manifest written by acquisition, or stage coordinates).
2. Load tiles into a numpy array; pass to ``m2stitch.stitch_images`` with the
   row/col indices (and optional ``position_initial_guess`` from stage steps).
3. Normalize the returned ``x_pos``/``y_pos`` to non-negative canvas coordinates,
   size a blank canvas, and paste each tile with Pillow (optionally blend seams).
4. Save the composite to ``out_file`` and return a :class:`~yosegi.models.MosaicResult`.
"""

from __future__ import annotations

from pathlib import Path

from yosegi.models import MosaicResult


def stitch_tiles(in_dir: Path, out_file: Path) -> MosaicResult:
    """Align tiles in ``in_dir`` and write the merged mosaic to ``out_file``."""
    raise NotImplementedError(
        "Stitching is not implemented yet. This will align tiles with m2stitch and "
        "composite them onto a canvas with Pillow to produce a seamless mosaic."
    )
