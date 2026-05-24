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


def _shift_ncc(a, b, axis: int):
    """Best (ncc, shift) of ``b`` relative to ``a`` along ``axis`` (0=rows/y, 1=cols/x).

    A small 1-D sliding normalized cross-correlation over the overlap region.
    Returns ``(best_ncc, best_shift_px)``; the shift is always positive.
    """
    size = a.shape[axis]
    best = (-1.0, 0)
    for shift in range(size // 10, size - size // 10, 4):
        if axis == 1:
            o1, o2 = a[:, shift:], b[:, : size - shift]
        else:
            o1, o2 = a[shift:, :], b[: size - shift, :]
        if o1.size == 0:
            continue
        x = o1.astype(float)
        y = o2.astype(float)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        ncc = float((x * y).mean())
        if ncc > best[0]:
            best = (ncc, shift)
    return best


def _measure_steps_per_pixel(scope: Microscope, probe_steps: int = 3000, min_ncc: float = 0.3):
    """Estimate stage steps-per-pixel for each axis with one move+measure per axis.

    Captures a frame, moves ``probe_steps`` along one axis, captures again, and
    finds the pixel shift by cross-correlation; steps-per-pixel is
    ``probe_steps / shift``. Returns ``{"x": spp_x, "y": spp_y}`` with ``None``
    for any axis whose correlation is too weak to trust. Best-effort: never
    raises, so a flaky measurement degrades to ``None`` rather than aborting a scan.
    """
    import numpy as np

    def grab():
        return np.asarray(scope.capture_image().convert("L"))

    result: dict[str, float | None] = {"x": None, "y": None}
    for axis_key, axis_idx, move in (("x", 1, {"x": probe_steps, "y": 0, "z": 0}),
                                     ("y", 0, {"x": 0, "y": probe_steps, "z": 0})):
        try:
            before = grab()
            scope.move_rel(move)
            after = grab()
            scope.move_rel({k: -v for k, v in move.items()})
            ncc, shift = _shift_ncc(before, after, axis_idx)
            if ncc >= min_ncc and shift > 0:
                result[axis_key] = round(probe_steps / shift, 3)
        except Exception:
            result[axis_key] = None
    return result


def fetch_tiles(
    host: str | None,
    out_dir: Path,
    rows: int,
    cols: int,
    step_x: int,
    step_y: int,
    autofocus: bool = False,
    overlap: float | None = None,
    calibrate: bool = True,
    *,
    client: Microscope | None = None,
) -> list[Tile]:
    """Raster a grid, capture one tile per cell, and save them to ``out_dir``.

    The stage moves ``step_x``/``step_y`` steps between adjacent tiles in a snake
    pattern. ``overlap`` is recorded in the manifest but does not affect motion.
    When ``autofocus`` is set, the scope refocuses before each capture. When
    ``calibrate`` is set (default), one extra move+measure per axis estimates the
    stage steps-per-pixel and records it in the manifest, which lets the stitcher
    place tiles directly from their coordinates. Pass ``client`` to use an
    already-connected microscope (mainly for testing);
    otherwise one is opened from ``host`` (or mDNS discovery when ``host`` is
    ``None``).

    Returns one :class:`~yosegi.models.Tile` per captured patch and writes a
    ``manifest.json`` alongside the images.
    """
    if rows < 1 or cols < 1:
        raise AcquisitionError("rows and cols must be >= 1")

    scope = client if client is not None else connect(host)

    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AcquisitionError(f"Could not create output directory {out_dir}: {exc}") from exc

    steps_per_pixel = _measure_steps_per_pixel(scope) if calibrate else {"x": None, "y": None}

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
        path = out_dir / f"tile_r{row:02d}_c{col:02d}.jpg"
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
    _write_manifest(out_dir, rows, cols, step_x, step_y, overlap, autofocus, start, steps_per_pixel, tiles)
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
    steps_per_pixel: dict[str, float | None],
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
        "steps_per_pixel": steps_per_pixel,
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
