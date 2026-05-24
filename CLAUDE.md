# CLAUDE.md

Guidance for working in this repository.

## Goal

`yosegi-scope` automates assembly of microscopy mosaics from the
[OpenFlexure microscope](https://openflexure.org/). Today it scans a sample over an
XY grid, fetching overlapping image tiles over the local network, then aligns and
merges them into one seamless composite — no manual stitching.

The intended trajectory (see [Direction](#direction)) is a self-driving
digital-pathology pipeline: image **post-processing** for cleaner scans → automatic
**whole-slide survey** (detect the sample boundary, no user-defined grid) →
**brightfield→fluorescence** deep-learning translation → a **Pydantic/FastAPI**
service → a web **front end**.

## Tech stack

- **Python 3.11** (pinned `>=3.11,<3.12`: `openflexure-microscope-client==0.1.8`
  pins `pandas==1.5.3` / `pillow==9.5.0`, which lack 3.12 wheels).
- **uv** — environment and dependency management.
- **Typer** — CLI. **openflexure-microscope-client** — scope control.
  **openflexure-stitching** — the official stitching engine (affine placement +
  high-pass correlation). **piexif** — write tile metadata to EXIF.
- **ruff** (lint, config in `pyproject.toml`) and **pytest** (tests).

> **System dependency:** `openflexure-stitching` uses `pyvips`, which needs the
> native **libvips** library. Install it with `brew install vips` (macOS) or
> `apt-get install libvips` (Debian/CI). On macOS, pyvips may not find the dylib;
> if `import openflexure_stitching` fails with a libvips load error, run with
> `DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix)/lib`.

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
- The camera's image axes are **rotated ~90°** relative to the stage. This is
  captured exactly by the scope's **camera-stage-mapping (CSM) affine matrix**
  (e.g. `[[0.01, -4.40], [-4.37, 0.0]]` — large off-diagonal, ~4.4 steps/px), which
  `acquire` reads from the scope and embeds in each tile. The affine handles the
  rotation and X/Y scaling, so no manual axis-flipping is needed.
- Faint/low-texture samples correlate weakly, so correlation can disconnect tiles;
  stage+affine placement (`--no-correlate`) is the reliable fallback.

## Capture stats

- **Tile (frame): 832 × 624 px**, RGB JPEG, **~80 KB** each.
- A 3×3 scan → 9 tiles → ~2160 × 1620 px mosaic.
- **Capture order: boustrophedon (snake)** — row 0 left→right, row 1 right→left,
  etc., to minimise stage travel/backlash (`snake_cells` in `acquire.py`). Tiles
  keep true grid `(row, col)` indices regardless of visit order.
- The stage returns to its start position after a scan.

## How acquisition works

`acquire` reads the scope's CSM affine matrix from settings (running
`calibrate_xy()` if none is stored), then rasters the grid: `move_rel` by
`step_x`/`step_y` between tiles, optional `autofocus()` (on by default),
`capture_image()`, save as `tile_r{NN}_c{NN}.jpg`. **Each tile gets its stage
position and the CSM matrix written into EXIF UserComment** (raw UTF-8 JSON, the
format `openflexure-stitching` reads). A `manifest.json` (schema
`yosegi.acquire/1`) is also written as provenance.

## How stitching works

`stitch_tiles` calls `openflexure_stitching.load_tile_and_stitch`, which reads each
tile's stage position + CSM from EXIF, places tiles by `stage × affine`, and (when
`correlate=True`, the default) refines with high-pass phase correlation + a
least-squares global optimisation. `--no-correlate` does stage+affine placement
only — reliable on faint samples where correlation disconnects tiles. The library
writes its outputs into `out_file`'s parent; the full stitched image
(`*_stitched.jpg` or `stitched_from_stage.jpg`) is moved to `out_file`. Seam/
exposure blending is left to the library.

## CLI

```bash
uv run yosegi acquire --host <ip> -o ./tiles --rows 3 --cols 3 --step-x 2000 --step-y 2000
uv run yosegi stitch  -i ./tiles -o mosaic.jpg [--no-correlate --high-pass-sigma 5]
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
result = stitch_tiles("tiles", "mosaic.jpg")           # correlate=False for stage-only
```

`fetch_tiles` accepts a `client` (anything matching the `Microscope` Protocol),
which is how tests run without hardware.

## Working agreements

- **Before completing any feature, both must pass:** `uv run ruff check` and
  `uv run pytest`. CI enforces this on every PR (`.github/workflows/ci.yml`).
- Keep changes small with incremental, single-sentence commits.
- Match the existing style: `from __future__ import annotations`, lazy-import heavy
  libs (PIL / openflexure-stitching) inside functions, normalise failures into
  `AcquisitionError` / `StitchError`, validate before side effects.
- Tests must not require a microscope — fake the `Microscope` Protocol. Stitch
  tests that need libvips are skipped when `openflexure_stitching` can't import.

## Direction

Where the project is headed (each is a separate future effort, not yet built):

1. **Post-processing** *(next)* — flat-field/illumination correction, seam exposure
   blending, white-balance/contrast normalisation, optional denoising. Likely a new
   `postprocess.py` step applied to (or within) the stitch output.
2. **Automatic whole-slide survey** — detect the **sample boundary** (low-mag
   overview + thresholding/segmentation: tissue vs empty slide), then plan a scan
   that covers the whole sample with overlap. Replaces the user-supplied `--rows`/
   `--cols` grid in `acquire`.
3. **Brightfield → fluorescence** — a deep-learning model for virtual staining /
   modality translation on the mosaic. New inference module + model dependency.
4. **API** — wrap acquire/stitch/postprocess/survey/inference behind **FastAPI +
   Pydantic** models for programmatic and remote control.
5. **Front end** — web UI over the API to launch scans, watch progress, and
   browse/zoom mosaics.

When picking up any of these, keep the existing conventions (Protocol-based DI,
lazy heavy imports, `*Error` normalisation, ruff+pytest before merge).
