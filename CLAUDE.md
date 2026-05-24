# CLAUDE.md

Guidance for working in this repository.

## Goal

`yosegi-scope` automates assembly of microscopy mosaics from the
[OpenFlexure microscope](https://openflexure.org/). It scans a sample over an XY
grid, fetching overlapping image tiles from the scope over the local network,
then aligns and merges them into one seamless composite — no manual stitching.

## Tech stack

- **Python 3.11** (pinned `>=3.11,<3.12`: `openflexure-microscope-client==0.1.8`
  pins `pandas==1.5.3` / `pillow==9.5.0`, which lack 3.12 wheels).
- **uv** — environment and dependency management.
- **Typer** — CLI. **Pillow** + **numpy** — imaging. **m2stitch** — optional
  correlation-based alignment. **openflexure-microscope-client** — scope control.
- **ruff** (lint, config in `pyproject.toml`) and **pytest** (tests).

## Layout (`src/` layout)

- `src/yosegi/acquire.py` — drive the scope, raster the grid, write tiles + manifest.
- `src/yosegi/stitch.py` — place tiles into a mosaic (coordinates by default).
- `src/yosegi/cli.py` — Typer app: `acquire`, `stitch`, `run`.
- `src/yosegi/models.py` — `Tile`, `MosaicResult` dataclasses.
- `tests/` — pytest; hardware is faked, so no scope is needed to run them.

## OpenFlexure microscope

A Raspberry-Pi-based microscope exposing a Web-of-Things HTTP API on port 5000.
We talk to it via the official `openflexure-microscope-client` (`MicroscopeClient`),
which gives: `position` (`{x,y,z}` stage **steps**), `move`/`move_rel`,
`capture_image()` (PIL image from the preview stream), and `autofocus()`. Connect
by host/IP, or discover on the LAN via mDNS (`find_first_microscope`).

**Scope-specific quirks (measured against a real unit):**
- Camera **image axes are inverted** vs the stage: increasing stage X moves image
  content left, increasing Y moves it up. Stitch negates both axes
  (`_STAGE_SIGN_X/Y` in `stitch.py`) so mosaics aren't mirrored.
- **~4.3–4.6 stage steps per pixel** in X, ~5.5 in Y (objective-dependent).
- Faint/low-texture samples correlate weakly (NCC often < 0.3), which is why
  coordinate placement — not correlation — is the default.

## Capture stats

- **Tile (frame): 832 × 624 px**, RGB JPEG, **~80 KB** each.
- A 3×3 scan → 9 tiles → ~2160 × 1620 px mosaic.
- **Capture order: boustrophedon (snake)** — row 0 left→right, row 1 right→left,
  etc., to minimise stage travel/backlash (`snake_cells` in `acquire.py`). Tiles
  keep true grid `(row, col)` indices regardless of visit order.
- The stage returns to its start position after a scan.

## How acquisition works

Each `acquire` (before scanning) does one move+measure per axis to estimate
steps-per-pixel, then rasters the grid: `move_rel` by `step_x`/`step_y` between
tiles, optional `autofocus()` (on by default), `capture_image()`, save as
`tile_r{NN}_c{NN}.jpg`. It writes `manifest.json` (schema `yosegi.acquire/1`) with
the grid, step sizes, `steps_per_pixel`, start position, and per-tile stage
coordinates — the handoff the stitcher consumes.

## How stitching works

Default: **coordinate placement** — convert each tile's stage coordinates to
pixels via `steps_per_pixel` and paste with Pillow. Robust, always coherent (it
trusts the motors, not image content). Falls back to a regular `overlap` grid when
calibration is missing. `--refine` additionally runs **m2stitch** (phase
correlation), seeded with the coordinate positions so it only searches a small
window; needs textured tiles, real overlap, and a ≥2×3 grid. Seam/exposure
blending is not yet implemented.

## CLI

```bash
uv run yosegi acquire --host <ip> -o ./tiles --rows 3 --cols 3 --step-x 2000 --step-y 2000
uv run yosegi stitch  -i ./tiles -o mosaic.jpg [--refine --ncc-threshold 0.3]
uv run yosegi run     --host <ip> -o mosaic.jpg          # acquire then stitch
```

Omit `--host` for mDNS auto-discovery. `--autofocus` is on by default
(`--no-autofocus` to skip). `--step-x`/`--step-y` are in **stage steps**;
`--overlap` is metadata only. Errors print a one-line `Error:` and exit 1 (no
traceback).

## Library

```python
from yosegi.acquire import fetch_tiles
from yosegi.stitch import stitch_tiles

tiles = fetch_tiles(host="192.168.1.50", out_dir="tiles", rows=3, cols=3,
                    step_x=2000, step_y=2000)          # pass client=<fake> in tests
result = stitch_tiles("tiles", "mosaic.jpg")           # refine=True to correlate
```

`fetch_tiles` accepts a `client` (anything matching the `Microscope` Protocol),
which is how tests run without hardware.

## Working agreements

- **Before completing any feature, both must pass:** `uv run ruff check` and
  `uv run pytest`. CI enforces this on every PR (`.github/workflows/ci.yml`).
- Keep changes small with incremental, single-sentence commits.
- Match the existing style: `from __future__ import annotations`, lazy-import heavy
  libs (numpy/m2stitch/PIL) inside functions, normalise failures into
  `AcquisitionError` / `StitchError`, validate before side effects.
- Tests must not require a microscope — fake the `Microscope` Protocol.
