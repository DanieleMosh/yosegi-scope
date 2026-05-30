"""Fetch overlapping image tiles from an OpenFlexure microscope over the network.

Acquisition rasters an XY grid in a boustrophedon (snake) order, capturing one
tile per cell with the official ``openflexure-microscope-client``. The distance
moved between tiles is given explicitly in stage steps (``step_x``/``step_y``);
the requested ``overlap`` is recorded as metadata only.

Each tile is saved with its stage position and the scope's camera-stage-mapping
(CSM) affine matrix embedded in EXIF, in the format ``openflexure-stitching``
reads. A ``manifest.json`` is also written as human-readable provenance.
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
    def pull_settings(self) -> dict[str, Any]: ...
    def calibrate_xy(self) -> Any: ...


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


_CSM_EXTENSION = "org.openflexure.camera_stage_mapping"


def _get_csm(scope: Microscope, calibrate: bool) -> list[list[float]] | None:
    """Return the scope's camera-stage-mapping (CSM) affine matrix.

    Reads the stored ``image_to_stage_displacement`` matrix from the scope
    settings. When ``calibrate`` is set and no matrix is stored, runs the scope's
    ``calibrate_xy()`` once (slow: ~2 min of stage motion) and re-reads it.
    Returns ``None`` (best-effort) if no calibration can be obtained, in which
    case stitching falls back to stage coordinates plus correlation.
    """

    def read() -> list[list[float]] | None:
        try:
            ext = scope.pull_settings().get("extensions", {}).get(_CSM_EXTENSION, {})
            matrix = ext.get("image_to_stage_displacement")
            return matrix if matrix else None
        except Exception:
            return None

    matrix = read()
    if matrix is None and calibrate:
        try:
            scope.calibrate_xy()
            matrix = read()
        except Exception:
            matrix = None
    return matrix


def _write_tile_exif(path: Path, position: dict[str, int], csm: list[list[float]] | None) -> None:
    """Embed stage position and CSM in the tile's EXIF UserComment as JSON.

    This is the format ``openflexure-stitching`` reads: ``["stage"]["position"]``
    for the stage coordinates and
    ``["camera_stage_mapping"]["image_to_stage_displacement_matrix"]`` for the
    affine matrix. The UserComment is raw UTF-8 JSON (no charset prefix), matching
    how the library decodes it. Best-effort: a failure here does not abort the scan.
    """
    import piexif

    usercomment: dict[str, Any] = {"stage": {"position": dict(position)}}
    if csm is not None:
        usercomment["camera_stage_mapping"] = {"image_to_stage_displacement_matrix": csm}
    try:
        exif = {"Exif": {piexif.ExifIFD.UserComment: json.dumps(usercomment).encode("utf-8")}}
        piexif.insert(piexif.dump(exif), str(path))
    except Exception:
        pass


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
    ``calibrate`` is set (default), the scope's camera-stage-mapping is run if it
    has none stored. Each tile is saved with its stage position and the CSM affine
    matrix in EXIF, which is what the stitcher reads. Pass ``client`` to use an
    already-connected microscope (mainly for testing); otherwise one is opened
    from ``host`` (or mDNS discovery when ``host`` is ``None``).

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

    csm = _get_csm(scope, calibrate)

    start = dict(scope.position)
    # Translate the snake-grid into the absolute (row, col, x, y) plan that
    # the position-list helper consumes -- step relative to ``start`` so the
    # grid is anchored at the scope's current stage position.
    plan: list[tuple[int, int, int, int]] = [
        (r, c, start["x"] + c * step_x, start["y"] + r * step_y)
        for r, c in snake_cells(rows, cols)
    ]
    tiles = _capture_at_plan(scope, out_dir, plan, autofocus=autofocus, csm=csm)

    scope.move(start, absolute=True)
    _write_manifest(out_dir, rows, cols, step_x, step_y, overlap, autofocus, start, csm, tiles)
    return tiles


def fetch_tiles_at_positions(
    client: Microscope,
    out_dir: Path,
    positions: list[tuple[int, int]],
    *,
    rows: int,
    cols: int,
    autofocus: bool = False,
    calibrate: bool = False,
) -> list[Tile]:
    """Capture one tile at each absolute stage ``(x, y)`` in ``positions``.

    Used by the auto-survey pipeline to execute a :class:`~yosegi.survey.ScanPlan`
    that was computed from a detected bounding box. ``positions`` is assumed to
    cover a ``rows x cols`` snake-ordered grid (the planner's output); the
    ``(row, col)`` for each position is derived from its index so EXIF and
    filenames stay consistent with ``fetch_tiles``. The scope returns to its
    starting position when done. Pass ``calibrate=True`` to run camera-stage
    mapping when the scope has none stored.
    """
    if rows < 1 or cols < 1:
        raise AcquisitionError("rows and cols must be >= 1")
    if len(positions) != rows * cols:
        raise AcquisitionError(
            f"positions has {len(positions)} entries but rows*cols = {rows * cols}"
        )

    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AcquisitionError(f"Could not create output directory {out_dir}: {exc}") from exc

    csm = _get_csm(client, calibrate)
    start = dict(client.position)

    snake = list(snake_cells(rows, cols))
    plan = [(r, c, x, y) for (r, c), (x, y) in zip(snake, positions, strict=True)]
    tiles = _capture_at_plan(client, out_dir, plan, autofocus=autofocus, csm=csm)
    client.move(start, absolute=True)
    return tiles


def _capture_at_plan(
    scope: Microscope,
    out_dir: Path,
    plan: list[tuple[int, int, int, int]],
    *,
    autofocus: bool,
    csm: list[list[float]] | None,
) -> list[Tile]:
    """Walk ``plan`` of ``(row, col, abs_x, abs_y)`` and capture one tile per entry.

    Uses ``move_rel`` between steps so the scope incurs only the per-step travel,
    not the cumulative distance from the origin. Shared by both ``fetch_tiles``
    (regular snake grid) and ``fetch_tiles_at_positions`` (planned scan).
    """
    tiles: list[Tile] = []
    prev: tuple[int, int] | None = None
    for row, col, abs_x, abs_y in plan:
        if prev is not None:
            dx = abs_x - prev[0]
            dy = abs_y - prev[1]
            if dx or dy:
                scope.move_rel({"x": dx, "y": dy, "z": 0})
        if autofocus:
            scope.autofocus()
        image = scope.capture_image()
        path = out_dir / f"tile_r{row:02d}_c{col:02d}.jpg"
        image.save(path)
        pos = dict(scope.position)
        _write_tile_exif(path, pos, csm)
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
        prev = (abs_x, abs_y)
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
    csm: list[list[float]] | None,
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
        "camera_stage_mapping": csm,
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
