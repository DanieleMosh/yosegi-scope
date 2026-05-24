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


def stitch_tiles(in_dir: Path, out_file: Path) -> MosaicResult:
    """Align tiles in ``in_dir`` and write the merged mosaic to ``out_file``."""
    _discover_tiles(Path(in_dir))
    raise NotImplementedError(
        "Stitching is not implemented yet. This will align tiles with m2stitch and "
        "composite them onto a canvas with Pillow to produce a seamless mosaic."
    )
