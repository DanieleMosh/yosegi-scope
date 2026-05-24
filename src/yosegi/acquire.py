"""Fetch overlapping image tiles from an OpenFlexure microscope over the network.

Acquisition rasters an XY grid in a boustrophedon (snake) order, capturing one
tile per cell with the official ``openflexure-microscope-client``. The distance
moved between tiles is given explicitly in stage steps (``step_x``/``step_y``);
the requested ``overlap`` is recorded as metadata only. Each run writes a
``manifest.json`` describing the grid and per-tile stage positions, which the
stitching step consumes to reconstruct the mosaic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from yosegi import __version__
from yosegi.models import Tile

_ALLOWED_FORMATS = {"jpg", "jpeg", "png"}


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
    step_x: int,
    step_y: int,
    autofocus: bool = False,
    overlap: float | None = None,
    *,
    client: Microscope | None = None,
    image_format: str = "jpg",
) -> list[Tile]:
    """Raster a grid, capture one tile per cell, and save them to ``out_dir``.

    The stage moves ``step_x``/``step_y`` steps between adjacent tiles in a snake
    pattern. ``overlap`` is recorded in the manifest but does not affect motion.
    When ``autofocus`` is set, the scope refocuses before each capture. Pass
    ``client`` to use an already-connected microscope (mainly for testing);
    otherwise one is opened from ``host`` (or mDNS discovery when ``host`` is
    ``None``).

    Returns one :class:`~yosegi.models.Tile` per captured patch and writes a
    ``manifest.json`` alongside the images.
    """
    if rows < 1 or cols < 1:
        raise AcquisitionError("rows and cols must be >= 1")
    ext = image_format.lower()
    if ext not in _ALLOWED_FORMATS:
        raise AcquisitionError(f"image_format must be one of {sorted(_ALLOWED_FORMATS)}, got {image_format!r}")

    scope = client if client is not None else connect(host)

    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AcquisitionError(f"Could not create output directory {out_dir}: {exc}") from exc

    start = dict(scope.position)
    tiles: list[Tile] = []
    prev: tuple[int, int] | None = None

    for row, col in snake_cells(rows, cols):
        if prev is not None:
            dx = (col - prev[1]) * step_x
            dy = (row - prev[0]) * step_y
            if dx or dy:
                scope.move_rel({"x": dx, "y": dy, "z": 0})
        if autofocus:
            scope.autofocus()
        image = scope.capture_image()
        path = out_dir / f"tile_r{row:02d}_c{col:02d}.{ext}"
        image.save(path)
        pos = dict(scope.position)
        tiles.append(
            Tile(
                path=path,
                row=row,
                col=col,
                stage_x=pos.get("x"),
                stage_y=pos.get("y"),
                stage_z=pos.get("z"),
            )
        )
        prev = (row, col)

    scope.move(start, absolute=True)
    _write_manifest(out_dir, rows, cols, step_x, step_y, overlap, autofocus, start, tiles)
    return tiles


def _write_manifest(
    out_dir: Path,
    rows: int,
    cols: int,
    step_x: int,
    step_y: int,
    overlap: float | None,
    autofocus: bool,
    start: dict[str, int],
    tiles: list[Tile],
) -> Path:
    """Write the acquire->stitch handoff manifest to ``out_dir/manifest.json``."""
    manifest = {
        "schema": "yosegi.acquire/1",
        "tool_version": __version__,
        "grid": {"rows": rows, "cols": cols},
        "step": {"x": step_x, "y": step_y},
        "overlap": overlap,
        "autofocus": autofocus,
        "start_position": start,
        "tiles": [
            {
                "filename": t.path.name,
                "row": t.row,
                "col": t.col,
                "stage_x": t.stage_x,
                "stage_y": t.stage_y,
                "stage_z": t.stage_z,
            }
            for t in tiles
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
