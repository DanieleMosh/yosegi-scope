# yosegi-scope

Automated assembly of microscopy mosaics captured by the
[OpenFlexure microscope](https://openflexure.org/).

As the scope scans a sample it produces a series of overlapping image patches.
`yosegi-scope` fetches those tiles from the microscope over the local network,
aligns them, and merges them into a single seamless composite — no manual
stitching required.

> **Status:** acquisition and stitching both work — `acquire` rasters a grid and
> saves tiles + manifest, and `stitch` aligns and merges them into a composite.

## How it works

- **Acquire** — [`openflexure-microscope-client`](https://pypi.org/project/openflexure-microscope-client/)
  drives the scope (mDNS/IP discovery, stage moves, autofocus, capture) to raster
  an XY grid of overlapping tiles.
- **Stitch** — by default tiles are placed from their recorded **stage
  coordinates**, converted to pixels with a steps-per-pixel calibration that
  `acquire` measures automatically and writes into `manifest.json`. The
  OpenFlexure camera's axis orientation (image inverted vs the stage) is applied
  automatically, so the mosaic comes out the right way round with no flags. This is
  robust and always produces a coherent mosaic (it trusts the motors, not image
  content), then [Pillow](https://python-pillow.org/) composites the tiles onto one
  canvas.
  Pass `--refine` to additionally run [`m2stitch`](https://m2stitch.readthedocs.io/)
  (MIST-based phase correlation) for pixel-perfect seams — it is seeded with the
  coordinate positions, so it only searches a small window. Refinement needs
  textured tiles, real overlap, and a ≥2×3 grid; `--ncc-threshold` (lower for faint
  samples) and `--transpose` (the OpenFlexure camera's image axes are swapped vs the
  stage) tune it. The grid layout is read from `manifest.json`, or recovered from the
  `tile_r{row}_c{col}` filenames. Seam/exposure blending is a future enhancement.

## Requirements

- Python **3.11** (pinned: some pinned dependency wheels are unavailable on 3.12).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

## Install

```bash
uv sync
```

## Usage

```bash
# Show all commands
uv run yosegi --help

# Scan a 3x3 grid and save overlapping tiles (2000 stage steps between tiles)
uv run yosegi acquire --host microscope.local --output ./tiles \
    --rows 3 --cols 3 --step-x 2000 --step-y 2000

# Autofocus runs at every tile by default; pass --no-autofocus to skip it
uv run yosegi acquire --host microscope.local --output ./tiles --no-autofocus

# Stitch a folder of tiles into one composite (placed by stage coordinates)
uv run yosegi stitch --input ./tiles --output mosaic.jpg

# Refine the seams with m2stitch correlation (faint sample needs a lower threshold)
uv run yosegi stitch --input ./tiles --output mosaic.jpg --refine --ncc-threshold 0.3

# Acquire then stitch in one pass
uv run yosegi run --host microscope.local --output mosaic.jpg
```

If `--host` is omitted, the microscope is discovered automatically via mDNS.

The stage moves `--step-x`/`--step-y` **stage steps** between adjacent tiles, in a
snake pattern, autofocuses at each tile (disable with `--no-autofocus`), and returns
to the start when done. The right step size depends on your objective and sample —
pick a value that leaves the desired overlap between neighbouring tiles. Before
scanning, `acquire` measures the stage **steps-per-pixel**
(one move+measure per axis) and records it in `manifest.json` along with the grid and
per-tile stage positions; this is what lets the stitcher place tiles by coordinate.
`--overlap` is recorded as metadata and is only used as a placement fallback when
calibration is unavailable.

## Development

```bash
uv run pytest        # tests
uv run ruff check    # lint
```

## Roadmap

- [x] Implement the acquisition raster (XY grid, autofocus, capture) in `acquire.py`.
- [x] Implement m2stitch alignment + Pillow compositing in `stitch.py`.
- [ ] Seam blending and exposure/flat-field correction.
- [ ] Config file and richer error handling.
- [ ] CI workflow.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
