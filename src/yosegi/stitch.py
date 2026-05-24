"""Align overlapping tiles and merge them into one seamless composite.

This wraps the official ``openflexure-stitching`` library, which is purpose-built
for the OpenFlexure microscope: it reads each tile's stage position and the
camera-stage-mapping (CSM) affine matrix from EXIF (written by ``acquire``),
places tiles from those coordinates, and refines with high-pass-filtered phase
correlation plus a least-squares global optimisation.

Tiles must carry the metadata ``acquire`` embeds (stage position + CSM in EXIF).
The CSM is also read from ``manifest.json`` as a fallback. Failures are
normalized into :class:`StitchError`.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from yosegi.models import MosaicResult

_MANIFEST_SCHEMA = "yosegi.acquire/1"
# Tile files written by acquire, e.g. tile_r00_c01.jpg. Used to count input tiles
# without picking up the diagnostic PNGs openflexure-stitching drops in the folder.
_TILE_RE = re.compile(r"^tile_r\d+_c\d+\.(jpe?g|png)$", re.IGNORECASE)


class StitchError(RuntimeError):
    """Raised when tiles cannot be discovered, loaded, or aligned into a mosaic."""


def _csm_from_manifest(in_dir: Path) -> list[list[float]] | None:
    """Return the CSM affine matrix from ``in_dir/manifest.json`` if present."""
    manifest_path = in_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema") != _MANIFEST_SCHEMA:
        return None
    return manifest.get("camera_stage_mapping")


def stitch_tiles(
    in_dir: Path,
    out_file: Path,
    correlate: bool = True,
    high_pass_sigma: float = 10.0,
    minimum_overlap: float = 0.2,
) -> MosaicResult:
    """Stitch the tiles in ``in_dir`` and write the mosaic to ``out_file``.

    Uses ``openflexure-stitching``: tiles are placed from their EXIF stage
    coordinates and CSM affine matrix, then (when ``correlate`` is set) refined by
    high-pass phase correlation. ``high_pass_sigma`` and ``minimum_overlap`` tune
    the correlation. With ``correlate=False`` the placement is stage-coordinates
    only (fast, no correlation). The library writes its outputs (preview, tile
    config) into ``out_file``'s parent; the full stitched image is moved to
    ``out_file``. Raises :class:`StitchError` (writing nothing useful) on failure.
    """
    in_dir = Path(in_dir)
    out_file = Path(out_file)
    if not in_dir.is_dir():
        raise StitchError(f"Input directory does not exist: {in_dir}")

    try:
        from openflexure_stitching import (
            CorrelationSettings,
            LoadingSettings,
            OutputSettings,
            load_tile_and_stitch,
        )
    except Exception as exc:  # missing libvips / import failure
        raise StitchError(
            f"openflexure-stitching is unavailable ({exc}). Is libvips installed "
            f"(brew install vips / apt-get install libvips)?"
        ) from exc

    out_dir = out_file.parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StitchError(f"Could not create output directory {out_dir}: {exc}") from exc

    loading = LoadingSettings(csm_matrix=_csm_from_manifest(in_dir))
    correlation = CorrelationSettings(high_pass_sigma=high_pass_sigma, minimum_overlap=minimum_overlap)
    output = OutputSettings(
        output_dir=str(out_dir),
        stitching_mode="all" if correlate else "stage_stitch",
    )

    started_at = time.time()
    try:
        load_tile_and_stitch(
            str(in_dir),
            loading_settings=loading,
            correlation_settings=correlation,
            output_settings=output,
        )
    except Exception as exc:
        raise StitchError(
            f"Could not stitch tiles ({type(exc).__name__}: {exc}). Check that tiles "
            f"carry stage positions in EXIF and overlap enough; try --high-pass-sigma."
        ) from exc

    produced = _find_stitched_image(out_dir, since=started_at)
    if produced is None:
        raise StitchError(f"Stitching produced no output image in {out_dir}")
    if produced != out_file:
        shutil.move(str(produced), str(out_file))

    from PIL import Image

    with Image.open(out_file) as img:
        width, height = img.size
    tile_count = sum(1 for p in in_dir.iterdir() if _TILE_RE.match(p.name))
    return MosaicResult(path=out_file, width=width, height=height, tile_count=tile_count)


def _find_stitched_image(out_dir: Path, since: float) -> Path | None:
    """Return the full stitched image the current run wrote, if any.

    Correlated stitching writes ``{prefix}_stitched.jpg``; stage-only stitching
    writes ``stitched_from_stage.jpg``. Only files modified at or after ``since``
    are considered, so a stale image from a previous run is never picked up; the
    most recently modified match wins.
    """
    candidates = [
        p
        for p in [out_dir / "stitched_from_stage.jpg", *out_dir.glob("*_stitched.jpg")]
        if p.exists() and p.stat().st_mtime >= since
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
