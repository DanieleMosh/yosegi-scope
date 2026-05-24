"""Tests for tile stitching, using synthetic textured tiles (no hardware)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yosegi.stitch import StitchError, stitch_tiles


def _textured_source(h: int, w: int, seed: int = 1234) -> Image.Image:
    """High-contrast blobs give strong phase correlation (NCC ~0.8). Deterministic."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float64)
    n = int(0.012 * h * w)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    brightness = rng.integers(120, 256, n)
    radii = rng.integers(2, 6, n)
    yy, xx = np.mgrid[0:h, 0:w]
    for y, x, b, rd in zip(ys, xs, brightness, radii):
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= rd * rd
        img[mask] = np.maximum(img[mask], b)
    return Image.fromarray(img.astype(np.uint8)).convert("RGB")


def _make_grid(
    d: Path,
    rows: int,
    cols: int,
    tile: int = 200,
    overlap: float = 0.4,
    seed: int = 1234,
    *,
    manifest: bool = True,
) -> tuple[int, int]:
    """Write an overlapping tile grid (+optional manifest). Returns (src_w, src_h)."""
    d.mkdir(parents=True, exist_ok=True)
    step = int(round(tile * (1 - overlap)))
    src_w = step * (cols - 1) + tile
    src_h = step * (rows - 1) + tile
    src = _textured_source(src_h, src_w, seed)
    entries = []
    for r in range(rows):
        for c in range(cols):
            box = (c * step, r * step, c * step + tile, r * step + tile)
            name = f"tile_r{r:02d}_c{c:02d}.png"
            src.crop(box).save(d / name)
            entries.append(
                {"filename": name, "row": r, "col": c, "stage_x": c * 100, "stage_y": r * 100, "stage_z": 0}
            )
    if manifest:
        (d / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "yosegi.acquire/1",
                    "tool_version": "0.1.0",
                    "grid": {"rows": rows, "cols": cols},
                    "step": {"x": 100, "y": 100},
                    "overlap": overlap,
                    "autofocus": False,
                    "start_position": {"x": 0, "y": 0, "z": 0},
                    "tiles": entries,
                }
            )
        )
    return src_w, src_h


def test_stitch_from_manifest(tmp_path: Path) -> None:
    src_w, src_h = _make_grid(tmp_path, rows=2, cols=3)
    out = tmp_path / "mosaic.png"
    result = stitch_tiles(in_dir=tmp_path, out_file=out)

    assert out.exists()
    assert result.tile_count == 6
    assert result.path == out
    # m2stitch reconstructs the source size within one tile's tolerance
    assert abs(result.width - src_w) <= 200
    assert abs(result.height - src_h) <= 200
    assert Image.open(out).size == (result.width, result.height)


def test_stitch_with_lower_ncc_threshold(tmp_path: Path) -> None:
    # a permissive threshold still stitches the clean synthetic grid
    _make_grid(tmp_path, rows=2, cols=3)
    out = tmp_path / "mosaic.png"
    result = stitch_tiles(in_dir=tmp_path, out_file=out, ncc_threshold=0.3)
    assert out.exists()
    assert result.tile_count == 6


def test_stitch_from_filename_glob_fallback(tmp_path: Path) -> None:
    _make_grid(tmp_path, rows=2, cols=3, manifest=False)
    assert not (tmp_path / "manifest.json").exists()
    out = tmp_path / "mosaic.png"
    result = stitch_tiles(in_dir=tmp_path, out_file=out)
    assert result.tile_count == 6
    assert out.exists()


def test_creates_missing_output_parent(tmp_path: Path) -> None:
    _make_grid(tmp_path, rows=2, cols=3)
    out = tmp_path / "nested" / "deep" / "mosaic.png"
    result = stitch_tiles(in_dir=tmp_path, out_file=out)
    assert out.exists()
    assert result.path == out


def test_one_dimensional_grid_raises(tmp_path: Path) -> None:
    _make_grid(tmp_path, rows=1, cols=3)  # no top pairs
    with pytest.raises(StitchError):
        stitch_tiles(in_dir=tmp_path, out_file=tmp_path / "m.png")
    assert not (tmp_path / "m.png").exists()  # nothing written on failure


def test_mismatched_tile_sizes_raise(tmp_path: Path) -> None:
    _make_grid(tmp_path, rows=2, cols=3, manifest=False)
    Image.new("RGB", (123, 77)).save(tmp_path / "tile_r00_c00.png")
    with pytest.raises(StitchError):
        stitch_tiles(in_dir=tmp_path, out_file=tmp_path / "m.png")


def test_no_tiles_raises(tmp_path: Path) -> None:
    with pytest.raises(StitchError):
        stitch_tiles(in_dir=tmp_path, out_file=tmp_path / "m.png")


def test_missing_input_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(StitchError):
        stitch_tiles(in_dir=tmp_path / "nope", out_file=tmp_path / "m.png")


def test_manifest_references_missing_file_raises(tmp_path: Path) -> None:
    _make_grid(tmp_path, rows=2, cols=3)
    (tmp_path / "tile_r00_c00.png").unlink()  # manifest still lists it
    with pytest.raises(StitchError):
        stitch_tiles(in_dir=tmp_path, out_file=tmp_path / "m.png")


def test_bad_manifest_schema_raises(tmp_path: Path) -> None:
    _make_grid(tmp_path, rows=2, cols=3)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "wrong/9", "tiles": []}))
    with pytest.raises(StitchError):
        stitch_tiles(in_dir=tmp_path, out_file=tmp_path / "m.png")
