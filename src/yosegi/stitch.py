"""Align overlapping tiles and merge them into one seamless composite.

Tiles are aligned with ``m2stitch`` (a MIST-inspired phase-correlation grid
stitcher) and composited onto a single canvas with Pillow. The tile/grid layout
is read from the acquire manifest (schema ``yosegi.acquire/1``) when present, and
otherwise recovered from the ``tile_r{row}_c{col}`` filename convention.

m2stitch needs textured tiles, real overlap, and at least two tiles in each
direction; when it cannot align the grid the failure is normalized into
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


class StitchError(RuntimeError):
    """Raised when tiles cannot be discovered, loaded, or aligned into a mosaic."""


def _discover_tiles(in_dir: Path) -> list[tuple[Path, int, int]]:
    """Return ``[(path, row, col), ...]`` sorted by ``(row, col)``.

    Prefers ``manifest.json`` (schema ``yosegi.acquire/1``); falls back to
    globbing ``tile_r*_c*`` files and parsing row/col from the filename.
    """
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise StitchError(f"Input directory does not exist: {in_dir}")
    manifest_path = in_dir / "manifest.json"
    if manifest_path.exists():
        return _tiles_from_manifest(in_dir, manifest_path)
    return _tiles_from_glob(in_dir)


def _tiles_from_manifest(in_dir: Path, manifest_path: Path) -> list[tuple[Path, int, int]]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StitchError(f"Could not read manifest {manifest_path}: {exc}") from exc
    schema = manifest.get("schema")
    if schema != _MANIFEST_SCHEMA:
        raise StitchError(
            f"Unsupported manifest schema {schema!r} in {manifest_path} (expected {_MANIFEST_SCHEMA!r})"
        )
    tiles: list[tuple[Path, int, int]] = []
    for entry in manifest.get("tiles") or []:
        try:
            path = in_dir / entry["filename"]
            row = int(entry["row"])
            col = int(entry["col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StitchError(f"Malformed tile entry in manifest: {entry!r} ({exc})") from exc
        if not path.exists():
            raise StitchError(f"Manifest references missing tile file: {path}")
        tiles.append((path, row, col))
    if not tiles:
        raise StitchError(f"Manifest {manifest_path} lists no tiles")
    tiles.sort(key=lambda t: (t[1], t[2]))
    return tiles


def _tiles_from_glob(in_dir: Path) -> list[tuple[Path, int, int]]:
    tiles: list[tuple[Path, int, int]] = []
    for path in sorted(in_dir.iterdir()):
        if not path.is_file():
            continue
        match = _TILE_RE.match(path.name)
        if not match:
            continue
        tiles.append((path, int(match.group(1)), int(match.group(2))))
    if not tiles:
        raise StitchError(
            f"No tiles found in {in_dir}. Expected a manifest.json or files named like tile_r00_c00.jpg"
        )
    tiles.sort(key=lambda t: (t[1], t[2]))
    return tiles


def _load_tiles(tiles: list[tuple[Path, int, int]]):
    """Load tiles as a ``(N, H, W)`` grayscale stack plus parallel RGB images.

    All tiles must share one ``(H, W)``; m2stitch requires it. Returns
    ``(stack, rgb_images, rows, cols, tile_w, tile_h)``.
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    gray: list = []
    rgb: list = []
    rows: list[int] = []
    cols: list[int] = []
    size: tuple[int, int] | None = None  # (W, H)
    for path, row, col in tiles:
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


def _align(stack, rows: list[int], cols: list[int]):
    """Run m2stitch and return parallel ``(x_pos, y_pos)`` pixel arrays.

    ``row_col_transpose=False`` maps ``x_pos`` to the column (horizontal) axis
    and ``y_pos`` to the row (vertical) axis, matching PIL's paste convention.
    Any m2stitch failure is normalized into :class:`StitchError`.
    """
    import m2stitch

    n_rows = len(set(rows))
    n_cols = len(set(cols))
    if len(rows) < 4 or n_rows < 2 or n_cols < 2:
        raise StitchError(
            f"Need at least a 2x2 grid of overlapping tiles to stitch (found {n_rows} row(s) x "
            f"{n_cols} col(s), {len(rows)} tile(s)). A 2x3 or larger grid with ~30-40% overlap is recommended."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_df, _ = m2stitch.stitch_images(stack, rows=rows, cols=cols, row_col_transpose=False)
    except Exception as exc:  # AssertionError ("no good pair"), ValueError, internal filters, etc.
        raise StitchError(
            f"Could not align tiles ({type(exc).__name__}: {exc}). This usually means too little "
            f"overlap, too few tiles, or low-texture images. Try a >=2x3 grid with ~30-40% overlap."
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


def stitch_tiles(in_dir: Path, out_file: Path) -> MosaicResult:
    """Align tiles in ``in_dir`` and write the merged mosaic to ``out_file``.

    Reads the acquire manifest (or the tile filenames) to recover the grid,
    aligns the tiles with m2stitch, composites them with Pillow, and saves the
    result. Raises :class:`StitchError` (writing nothing) if the tiles cannot be
    discovered, loaded, or aligned. Returns a :class:`~yosegi.models.MosaicResult`.
    """
    in_dir = Path(in_dir)
    out_file = Path(out_file)

    tiles = _discover_tiles(in_dir)
    stack, rgb, rows, cols, tile_w, tile_h = _load_tiles(tiles)
    x_pos, y_pos = _align(stack, rows, cols)
    canvas, width, height = _composite(rgb, x_pos, y_pos, tile_w, tile_h)

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_file)
    except OSError as exc:
        raise StitchError(f"Could not write mosaic to {out_file}: {exc}") from exc

    return MosaicResult(path=out_file, width=width, height=height, tile_count=len(tiles))
