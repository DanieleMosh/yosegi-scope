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
- **Stitch** — uses [`openflexure-stitching`](https://gitlab.com/openflexure/openflexure-stitching),
  the official OpenFlexure tool. `acquire` embeds each tile's **stage position** and
  the scope's **camera-stage-mapping (CSM) affine matrix** in EXIF; the stitcher
  places tiles by `stage × affine` (correctly handling the camera's rotation and
  scaling) and refines with high-pass phase correlation + a least-squares global
  optimisation. `--no-correlate` does stage+affine placement only, which is reliable
  on faint samples where correlation can't connect tiles.

## Requirements

- Python **3.11** (pinned: some pinned dependency wheels are unavailable on 3.12).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- **libvips** — a native library `openflexure-stitching` needs:
  `brew install vips` (macOS) or `sudo apt-get install libvips` (Debian/Ubuntu).
  On macOS, if stitching fails to load libvips, prefix commands with
  `DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix)/lib`.

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

# Stitch a folder of tiles into one composite (stage+affine placement + correlation)
uv run yosegi stitch --input ./tiles --output mosaic.jpg

# Stage+affine placement only (reliable on faint/low-texture samples)
uv run yosegi stitch --input ./tiles --output mosaic.jpg --no-correlate

# Acquire then stitch in one pass
uv run yosegi run --host microscope.local --output mosaic.jpg
```

If `--host` is omitted, the microscope is discovered automatically via mDNS.

The stage moves `--step-x`/`--step-y` **stage steps** between adjacent tiles, in a
snake pattern, autofocuses at each tile (disable with `--no-autofocus`), and returns
to the start when done. The right step size depends on your objective and sample —
pick a value that leaves the desired overlap between neighbouring tiles. Before
scanning, `acquire` reads the scope's **camera-stage-mapping** calibration (running
it once if absent) and writes each tile's stage position and that affine matrix into
EXIF, which is what the stitcher uses to place tiles. `--overlap` is recorded as
metadata only.

## Development

```bash
uv run pytest        # tests
uv run ruff check    # lint
```

## Roadmap

- [x] Implement the acquisition raster (XY grid, autofocus, capture) in `acquire.py`.
- [x] Stitch via `openflexure-stitching` (EXIF stage coords + CSM affine matrix).
- [x] CI workflow (ruff + pytest).
- [ ] Tune correlation for faint samples (high-pass) / seam blending.
- [ ] Config file and richer error handling.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
