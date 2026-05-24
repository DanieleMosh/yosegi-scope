# yosegi-scope

Automated assembly of microscopy mosaics captured by the
[OpenFlexure microscope](https://openflexure.org/).

As the scope scans a sample it produces a series of overlapping image patches.
`yosegi-scope` fetches those tiles from the microscope over the local network,
aligns them, and merges them into a single seamless composite — no manual
stitching required.

> **Status:** boilerplate. The CLI surface is in place; acquisition and stitching
> are documented stubs awaiting implementation (see the roadmap below).

## How it works

- **Acquire** — [`openflexure-microscope-client`](https://pypi.org/project/openflexure-microscope-client/)
  drives the scope (mDNS/IP discovery, stage moves, autofocus, capture) to raster
  an XY grid of overlapping tiles.
- **Stitch** — [`m2stitch`](https://m2stitch.readthedocs.io/) (MIST-based) computes
  refined per-tile offsets, then [Pillow](https://python-pillow.org/) composites the
  tiles onto a single canvas.

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

# Scan a sample and save overlapping tiles
uv run yosegi acquire --host microscope.local --output ./tiles --rows 3 --cols 3 --overlap 0.2

# Stitch a folder of tiles into one composite
uv run yosegi stitch --input ./tiles --output mosaic.jpg

# Acquire then stitch in one pass
uv run yosegi run --host microscope.local --output mosaic.jpg
```

If `--host` is omitted, the microscope is discovered automatically via mDNS.

## Development

```bash
uv run pytest        # tests
uv run ruff check    # lint
```

## Roadmap

- [ ] Implement the acquisition raster (XY grid, autofocus, capture) in `acquire.py`.
- [ ] Implement m2stitch alignment + Pillow compositing in `stitch.py`.
- [ ] Seam blending and exposure/flat-field correction.
- [ ] Config file and richer error handling.
- [ ] CI workflow.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
