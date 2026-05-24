"""Align overlapping tiles and merge them into one seamless composite.

By default tiles are placed from their recorded stage coordinates (converted to
pixels with the steps-per-pixel calibration written by ``acquire``), which is
robust and always produces a coherent mosaic. With ``refine=True`` the placement
is improved by ``m2stitch`` (a MIST-inspired phase-correlation stitcher), seeded
with the coordinate positions so it only has to find a small correction.

The tile/grid layout is read from the acquire manifest (schema
``yosegi.acquire/1``) when present, and otherwise recovered from the
``tile_r{row}_c{col}`` filename convention. Failures are normalized into
:class:`StitchError` and no image is written.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

from yosegi.models import MosaicResult

_ALLOWED_EXTS = (".jpg", ".jpeg", ".png")
_MANIFEST_SCHEMA = "yosegi.acquire/1"
# tile_r00_c01.jpg -> row=0, col=1
_TILE_RE = re.compile(r"^tile_r(\d+)_c(\d+)\.(jpe?g|png)$", re.IGNORECASE)

# One tile: (path, row, col, stage_x | None, stage_y | None)
Tile = tuple[Path, int, int, "int | None", "int | None"]


class StitchError(RuntimeError):
    """Raised when tiles cannot be discovered, loaded, or aligned into a mosaic."""


def _discover_tiles(in_dir: Path) -> tuple[list[Tile], dict]:
    """Return ``(tiles, meta)`` where tiles are sorted by ``(row, col)``.

    ``tiles`` is a list of ``(path, row, col, stage_x, stage_y)``. ``meta`` holds
    manifest-level fields used for placement (``steps_per_pixel``, ``step``,
    ``overlap``) and is empty when there is no manifest. Prefers ``manifest.json``
    (schema ``yosegi.acquire/1``); falls back to globbing ``tile_r*_c*`` files.
    """
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise StitchError(f"Input directory does not exist: {in_dir}")
    manifest_path = in_dir / "manifest.json"
    if manifest_path.exists():
        return _tiles_from_manifest(in_dir, manifest_path)
    return _tiles_from_glob(in_dir), {}


def _tiles_from_manifest(in_dir: Path, manifest_path: Path) -> tuple[list[Tile], dict]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StitchError(f"Could not read manifest {manifest_path}: {exc}") from exc
    schema = manifest.get("schema")
    if schema != _MANIFEST_SCHEMA:
        raise StitchError(
            f"Unsupported manifest schema {schema!r} in {manifest_path} (expected {_MANIFEST_SCHEMA!r})"
        )
    tiles: list[Tile] = []
    for entry in manifest.get("tiles") or []:
        try:
            path = in_dir / entry["filename"]
            row = int(entry["row"])
            col = int(entry["col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StitchError(f"Malformed tile entry in manifest: {entry!r} ({exc})") from exc
        if not path.exists():
            raise StitchError(f"Manifest references missing tile file: {path}")
        tiles.append((path, row, col, entry.get("stage_x"), entry.get("stage_y")))
    if not tiles:
        raise StitchError(f"Manifest {manifest_path} lists no tiles")
    tiles.sort(key=lambda t: (t[1], t[2]))
    meta = {
        "steps_per_pixel": manifest.get("steps_per_pixel"),
        "step": manifest.get("step"),
        "overlap": manifest.get("overlap"),
    }
    return tiles, meta


def _tiles_from_glob(in_dir: Path) -> list[Tile]:
    tiles: list[Tile] = []
    for path in sorted(in_dir.iterdir()):
        if not path.is_file():
            continue
        match = _TILE_RE.match(path.name)
        if not match:
            continue
        tiles.append((path, int(match.group(1)), int(match.group(2)), None, None))
    if not tiles:
        raise StitchError(
            f"No tiles found in {in_dir}. Expected a manifest.json or files named like tile_r00_c00.jpg"
        )
    tiles.sort(key=lambda t: (t[1], t[2]))
    return tiles


def _load_tiles(tiles: list[Tile]):
    """Load tiles as a ``(N, H, W)`` grayscale stack plus parallel RGB images.

    All tiles must share one ``(H, W)``. Returns
    ``(stack, rgb_images, rows, cols, tile_w, tile_h)``.
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    gray: list = []
    rgb: list = []
    rows: list[int] = []
    cols: list[int] = []
    size: tuple[int, int] | None = None  # (W, H)
    for path, row, col, _sx, _sy in tiles:
        try:
            img = Image.open(path)
            img.load()
        except (OSError, UnidentifiedImageError) as exc:
            raise StitchError(f"Could not read tile image {path}: {exc}") from exc
        if size is None:
            size = img.size
        elif img.size != size:
            raise StitchError(
                f"Tile {path.name} is {img.size} but expected {size}; all tiles must be the same size."
            )
        rgb.append(img.convert("RGB"))
        gray.append(np.asarray(img.convert("L")))
        rows.append(row)
        cols.append(col)

    stack = np.array(gray)  # (N, H, W); uniform size guaranteed above
    tile_w, tile_h = size  # PIL size is (W, H)
    return stack, rgb, rows, cols, tile_w, tile_h


def _coordinate_positions(tiles: list[Tile], meta: dict, rows: list[int], cols: list[int],
                          tile_w: int, tile_h: int):
    """Compute per-tile pixel ``(x_pos, y_pos)`` from stage coordinates.

    Uses the manifest's ``steps_per_pixel`` to convert each tile's stage_x/stage_y
    to pixels. Falls back to a regular grid from ``overlap`` (then a default 20%
    overlap) when calibration or stage coordinates are unavailable. Always
    succeeds, so this is the robust default placement.
    """
    spp = (meta or {}).get("steps_per_pixel") or {}
    spp_x, spp_y = spp.get("x"), spp.get("y")
    have_coords = all(t[3] is not None and t[4] is not None for t in tiles)

    if spp_x and spp_y and have_coords:
        x_pos = [t[3] / spp_x for t in tiles]
        y_pos = [t[4] / spp_y for t in tiles]
        return x_pos, y_pos

    # Fallback: regular grid spacing from the requested overlap.
    overlap = (meta or {}).get("overlap")
    if not isinstance(overlap, (int, float)) or not (0 <= overlap < 1):
        overlap = 0.2
    step_px_x = tile_w * (1 - overlap)
    step_px_y = tile_h * (1 - overlap)
    x_pos = [c * step_px_x for c in cols]
    y_pos = [r * step_px_y for r in rows]
    return x_pos, y_pos


def _refine(stack, rows: list[int], cols: list[int], x_pos, y_pos, tile_w: int, tile_h: int,
            ncc_threshold: float = 0.5, transpose: bool = False):
    """Refine coordinate positions with m2stitch, seeded by the initial guess.

    Passes the coordinate positions as ``position_initial_guess`` so m2stitch only
    searches a small window around each tile, avoiding the spurious large jumps it
    makes on low-texture or repetitive samples. Returns refined ``(x_pos, y_pos)``;
    raises :class:`StitchError` if m2stitch cannot align.
    """
    import numpy as np
    import m2stitch

    n_rows, n_cols = len(set(rows)), len(set(cols))
    if len(rows) < 4 or n_rows < 2 or n_cols < 2:
        raise StitchError(
            f"Refinement needs at least a 2x2 grid (found {n_rows}x{n_cols}, {len(rows)} tiles)."
        )
    # m2stitch initial guess is in (y, x) pixel order per tile.
    guess = np.array([[float(y), float(x)] for x, y in zip(x_pos, y_pos)])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_df, _ = m2stitch.stitch_images(
                stack, rows=rows, cols=cols, row_col_transpose=transpose,
                ncc_threshold=ncc_threshold, position_initial_guess=guess,
            )
    except Exception as exc:
        raise StitchError(
            f"Refinement failed ({type(exc).__name__}: {exc}). Try without --refine, "
            f"lower --ncc-threshold (currently {ncc_threshold}), or --transpose."
        ) from exc
    return grid_df["x_pos"].to_numpy(), grid_df["y_pos"].to_numpy()


def _composite(rgb_images, x_pos, y_pos, tile_w: int, tile_h: int):
    """Paste tiles onto a canvas at normalized ``(x, y)`` positions.

    ``x_pos``/``y_pos`` are absolute, possibly-negative pixel positions; the
    minimums are subtracted so the top-left tile sits at ``(0, 0)``. Returns
    ``(canvas, width, height)``.
    """
    from PIL import Image

    min_x, min_y = min(x_pos), min(y_pos)
    xs = [int(round(x - min_x)) for x in x_pos]
    ys = [int(round(y - min_y)) for y in y_pos]
    width = max(xs) + tile_w
    height = max(ys) + tile_h
    canvas = Image.new("RGB", (width, height))
    for img, x, y in zip(rgb_images, xs, ys):
        canvas.paste(img, (x, y))
    return canvas, width, height


def stitch_tiles(
    in_dir: Path,
    out_file: Path,
    refine: bool = False,
    ncc_threshold: float = 0.5,
    transpose: bool = False,
) -> MosaicResult:
    """Place tiles in ``in_dir`` and write the merged mosaic to ``out_file``.

    By default tiles are placed from their recorded stage coordinates (robust,
    always coherent). With ``refine=True`` the placement is improved by m2stitch,
    seeded with those coordinates; if refinement fails the call raises
    :class:`StitchError`. ``ncc_threshold`` and ``transpose`` only affect
    refinement. Raises :class:`StitchError` (writing nothing) if tiles cannot be
    discovered or loaded. Returns a :class:`~yosegi.models.MosaicResult`.
    """
    in_dir = Path(in_dir)
    out_file = Path(out_file)

    tiles, meta = _discover_tiles(in_dir)
    stack, rgb, rows, cols, tile_w, tile_h = _load_tiles(tiles)

    x_pos, y_pos = _coordinate_positions(tiles, meta, rows, cols, tile_w, tile_h)
    if refine:
        x_pos, y_pos = _refine(
            stack, rows, cols, x_pos, y_pos, tile_w, tile_h,
            ncc_threshold=ncc_threshold, transpose=transpose,
        )

    canvas, width, height = _composite(rgb, x_pos, y_pos, tile_w, tile_h)

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_file)
    except OSError as exc:
        raise StitchError(f"Could not write mosaic to {out_file}: {exc}") from exc

    return MosaicResult(path=out_file, width=width, height=height, tile_count=len(tiles))
