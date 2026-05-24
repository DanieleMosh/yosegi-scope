"""Tests for stitching via openflexure-stitching.

The full-stitch tests need the native ``libvips`` library (for ``pyvips``); they
are skipped when ``openflexure_stitching`` cannot be imported. The error-path
tests run everywhere.
"""

import json
from pathlib import Path

import piexif
import pytest
from PIL import Image

from yosegi.stitch import StitchError, stitch_tiles

# Skip the heavy tests if libvips / openflexure-stitching is unavailable.
# pyvips raises OSError (not ImportError) when libvips is missing, so guard broadly.
try:
    import openflexure_stitching  # noqa: F401

    _OFS_AVAILABLE = True
except Exception:
    _OFS_AVAILABLE = False

requires_ofs = pytest.mark.skipif(not _OFS_AVAILABLE, reason="requires libvips / openflexure-stitching")


def _write_tile(path: Path, stage_x: int, stage_y: int, size: tuple[int, int] = (320, 240)) -> None:
    """Write a tile with stage position embedded in EXIF (raw UTF-8 JSON UserComment)."""
    Image.new("RGB", size, (120, 180, 160)).save(path)
    uc = {"stage": {"position": {"x": stage_x, "y": stage_y, "z": 0}}}
    exif = {"Exif": {piexif.ExifIFD.UserComment: json.dumps(uc).encode("utf-8")}}
    piexif.insert(piexif.dump(exif), str(path))


def _make_grid(d: Path, rows: int, cols: int, tile: int = 320, step: int = 200) -> None:
    """Write an overlapping grid of EXIF-tagged tiles (stage coords on a regular grid)."""
    d.mkdir(parents=True, exist_ok=True)
    for r in range(rows):
        for c in range(cols):
            _write_tile(d / f"tile_r{r:02d}_c{c:02d}.jpg", stage_x=c * step, stage_y=r * step)


@requires_ofs
def test_stage_stitch_writes_mosaic(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    _make_grid(tiles_dir, rows=2, cols=2)
    # a diagnostic PNG like openflexure-stitching drops in the folder must not be
    # counted as a tile (regression for the tile_count over-count bug)
    Image.new("RGB", (10, 10)).save(tiles_dir / "stitching_inputs.png")
    out = tmp_path / "mosaic.jpg"
    result = stitch_tiles(in_dir=tiles_dir, out_file=out, correlate=False)
    assert out.exists()
    assert result.path == out
    assert result.tile_count == 4
    assert result.width > 0 and result.height > 0


def test_missing_input_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(StitchError):
        stitch_tiles(in_dir=tmp_path / "nope", out_file=tmp_path / "m.jpg")


def test_find_stitched_image_ignores_stale_output(tmp_path: Path) -> None:
    import os

    from yosegi.stitch import _find_stitched_image

    # a stitched image from a previous run, with an mtime well before this run
    stale = tmp_path / "old_stitched.jpg"
    stale.write_bytes(b"old")
    os.utime(stale, (1000, 1000))
    started_at = 2000.0
    # nothing new since started_at -> the stale file must not be picked up
    assert _find_stitched_image(tmp_path, since=started_at) is None
    # a fresh output (mtime after started_at) is found
    fresh = tmp_path / "new_stitched.jpg"
    fresh.write_bytes(b"new")
    os.utime(fresh, (3000, 3000))
    assert _find_stitched_image(tmp_path, since=started_at) == fresh
