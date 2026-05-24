"""Fetch overlapping image tiles from an OpenFlexure microscope over the network.

This is a documented stub. The real implementation will use the official
``openflexure-microscope-client`` library::

    import openflexure_microscope_client as ofm

    scope = ofm.MicroscopeClient(host) if host else ofm.find_first_microscope()
    start = scope.position  # {'x', 'y', 'z'} in stage steps

Sketch of the raster scan to fill in here:

1. Compute the XY step between tiles from the camera field of view and ``overlap``
   (step = field_of_view * (1 - overlap)), in stage steps.
2. For each (row, col) in a boustrophedon (snake) raster over ``rows`` x ``cols``:
   - ``scope.move({...})`` to the target XY (keep Z fixed, or ``scope.autofocus()``).
   - ``img = scope.capture_image()`` (PIL image) and save to ``out_dir``.
   - Record ``scope.position`` as the tile's stage coordinates.
3. Return the list of :class:`~yosegi.models.Tile`, which the stitcher consumes.
"""

from __future__ import annotations

from pathlib import Path

from yosegi.models import Tile


def fetch_tiles(
    host: str | None,
    out_dir: Path,
    rows: int,
    cols: int,
    overlap: float,
) -> list[Tile]:
    """Scan the sample and download overlapping tiles into ``out_dir``.

    Returns one :class:`~yosegi.models.Tile` per captured patch.
    """
    raise NotImplementedError(
        "Acquisition is not implemented yet. This will drive the OpenFlexure scope via "
        "openflexure-microscope-client to raster an XY grid and save overlapping tiles."
    )
