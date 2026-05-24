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
- **Stitch** — [`m2stitch`](https://m2stitch.readthedocs.io/) (MIST-based) computes
  refined per-tile offsets, then [Pillow](https://python-pillow.org/) composites the
  tiles onto a single canvas. Stitching needs textured tiles with real overlap and a
  grid of at least 2×3; the layout is read from the run's `manifest.json` (or the tile
  filenames). Seeding the aligner from stage steps is a future enhancement.

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

# Autofocus before every tile
uv run yosegi acquire --host microscope.local --output ./tiles --autofocus

# Stitch a folder of tiles into one composite
uv run yosegi stitch --input ./tiles --output mosaic.jpg

# Acquire then stitch in one pass
uv run yosegi run --host microscope.local --output mosaic.jpg
```

If `--host` is omitted, the microscope is discovered automatically via mDNS.

The stage moves `--step-x`/`--step-y` **stage steps** between adjacent tiles, in a
snake pattern, and returns to the start when done. The right step size depends on
your objective and sample — pick a value that leaves the desired overlap between
neighbouring tiles. `--overlap` is recorded in the run's `manifest.json` as metadata
for the stitcher; it does not affect stage motion. Each run writes `manifest.json`
alongside the tiles, capturing the grid and per-tile stage positions.

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
