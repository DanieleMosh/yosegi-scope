"""Fetch overlapping image tiles from an OpenFlexure microscope over the network.

Acquisition rasters an XY grid in a boustrophedon (snake) order, capturing one
tile per cell with the official ``openflexure-microscope-client``. The distance
moved between tiles is given explicitly in stage steps (``step_x``/``step_y``);
the requested ``overlap`` is recorded as metadata only. Each run writes a
``manifest.json`` describing the grid and per-tile stage positions, which the
stitching step consumes to reconstruct the mosaic.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from yosegi.models import Tile


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot proceed (bad parameters or scope connection)."""


class Microscope(Protocol):
    """Structural type for the bits of the OpenFlexure client we use.

    Lets the real ``MicroscopeClient`` and an in-memory test double be used
    interchangeably without importing the heavy client library.
    """

    @property
    def position(self) -> dict[str, int]: ...
    def move(self, position: dict[str, int], absolute: bool = True) -> Any: ...
    def move_rel(self, position: dict[str, int]) -> Any: ...
    def capture_image(self) -> Any: ...  # returns a PIL.Image
    def autofocus(self, dz: int = 2000) -> Any: ...


def connect(host: str | None) -> Microscope:
    """Connect to a microscope by host/IP, or discover one via mDNS.

    Any failure (network error, no microscope found, missing library) is
    normalized into :class:`AcquisitionError` so callers can show a clean
    message instead of a traceback.
    """
    try:
        from openflexure_microscope_client import (
            MicroscopeClient,
            find_first_microscope,
        )

        return MicroscopeClient(host) if host else find_first_microscope()
    except Exception as exc:  # requests errors, mDNS "no microscopes", import errors
        target = host or "auto-discovery (mDNS)"
        raise AcquisitionError(f"Could not connect to microscope via {target}: {exc}") from exc


def snake_cells(rows: int, cols: int) -> Iterator[tuple[int, int]]:
    """Yield ``(row, col)`` grid indices in boustrophedon (snake) order.

    Even rows run left-to-right, odd rows right-to-left, so the stage reverses
    direction once per row instead of jumping back to the start each time. The
    yielded ``col`` is always the true grid column (only the visit order
    reverses), so tiles keep real grid coordinates.
    """
    for row in range(rows):
        col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in col_range:
            yield row, col


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
