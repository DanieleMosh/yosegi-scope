#!/usr/bin/env python
"""End-to-end integration test against a REAL OpenFlexure microscope.

Unlike the pytest suite (which fakes the ``Microscope`` Protocol and never needs
hardware), this script drives an actual scope over the network through the whole
automatic whole-slide survey: overview -> multi-region tissue detection ->
tissue-gated plan -> (optional) per-region focus map -> high-res scan -> stitch.
It then validates the artifacts each stage should have produced and prints a
pass/fail report.

Run it by hand -- it is deliberately NOT wired into CI (CI has no scope):

    # mDNS auto-discovery (default), full-default auto survey with a focus map
    uv run python scripts/e2e_scope.py

    # explicit host, custom output dir, no focus map
    YOSEGI_SCOPE_HOST=192.168.178.120 uv run python scripts/e2e_scope.py \
        --out-dir /tmp/yosegi_e2e --no-focus-map

macOS libvips note: prefix with
``DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix)/lib`` if stitching can't load
libvips.

Exit code is 0 only if every validation check passes; non-zero otherwise, so it
can gate a manual release check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _fmt(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name}: {c.detail}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--host",
        default=os.environ.get("YOSEGI_SCOPE_HOST"),
        help="Microscope host/IP. Default: $YOSEGI_SCOPE_HOST, else mDNS auto-discovery.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("YOSEGI_E2E_OUT", "e2e_out")),
        help="Directory for the mosaic and intermediate artifacts (default: ./e2e_out).",
    )
    p.add_argument("--overview-rows", type=int, default=5)
    p.add_argument("--overview-cols", type=int, default=5)
    p.add_argument("--overview-step-x", type=int, default=2500)
    p.add_argument("--overview-step-y", type=int, default=2500)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--min-area-frac", type=float, default=0.005)
    p.add_argument(
        "--focus-map",
        dest="focus_map",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Build a per-region focus surface (default: on; use --no-focus-map to disable).",
    )
    p.add_argument(
        "--no-correlate",
        dest="correlate",
        action="store_false",
        help="Stage+affine placement only (skip phase correlation). More robust on faint samples.",
    )
    return p.parse_args(argv)


def validate(out_file: Path, *, expect_focus_map: bool) -> list[Check]:
    """Check the artifacts a successful auto survey must have written."""
    from PIL import Image

    parent = out_file.parent
    stem = out_file.stem
    overview_dir = parent / f"{stem}_overview"
    overview_jpg = parent / f"{stem}_overview.jpg"
    tiles_dir = parent / f"{stem}_tiles"
    checks: list[Check] = []

    # Overview pass produced tiles + a manifest with a real CSM.
    ov_manifest_path = overview_dir / "manifest.json"
    if ov_manifest_path.exists():
        ov = json.loads(ov_manifest_path.read_text())
        csm = ov.get("camera_stage_mapping")
        n_ov = len(ov.get("tiles", []))
        checks.append(Check("overview tiles", n_ov > 0, f"{n_ov} overview tiles captured"))
        checks.append(
            Check("overview CSM", bool(csm), f"CSM present: {bool(csm)}")
        )
    else:
        checks.append(Check("overview manifest", False, f"missing {ov_manifest_path}"))

    # Stitched overview canvas the detector ran on.
    if overview_jpg.exists():
        with Image.open(overview_jpg) as im:
            w, h = im.size
        checks.append(Check("overview canvas", w > 0 and h > 0, f"{w}x{h} overview.jpg"))
    else:
        checks.append(Check("overview canvas", False, f"missing {overview_jpg}"))

    # Tissue-gated high-res scan: a manifest, >=1 tile, focus_map flag as expected.
    tiles_manifest_path = tiles_dir / "manifest.json"
    if tiles_manifest_path.exists():
        tm = json.loads(tiles_manifest_path.read_text())
        n_tiles = len(tm.get("tiles", []))
        checks.append(Check("high-res tiles", n_tiles >= 1, f"{n_tiles} tissue-gated tiles"))
        # If a focus map was requested, the manifest must record it and tile Z must vary.
        fm_flag = tm.get("focus_map", False)
        checks.append(
            Check(
                "focus map recorded",
                fm_flag == expect_focus_map,
                f"manifest focus_map={fm_flag} (expected {expect_focus_map})",
            )
        )
        zs = [t.get("stage_z") for t in tm.get("tiles", [])]
        zs_ok = all(z is not None for z in zs) and len(zs) > 0
        checks.append(Check("tile Z present", zs_ok, f"{len(set(zs))} distinct Z over {len(zs)} tiles"))
    else:
        checks.append(Check("high-res manifest", False, f"missing {tiles_manifest_path}"))

    # Final mosaic.
    if out_file.exists():
        with Image.open(out_file) as im:
            w, h = im.size
        checks.append(Check("final mosaic", w > 0 and h > 0, f"{w}x{h} -> {out_file}"))
    else:
        checks.append(Check("final mosaic", False, f"missing {out_file}"))

    return checks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Imported here so --help works without the heavy deps / a scope.
    from yosegi.acquire import AcquisitionError, connect
    from yosegi.focus import FocusError
    from yosegi.stitch import StitchError
    from yosegi.survey import SurveyError, run_auto_survey

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "e2e_mosaic.jpg"

    target = args.host or "mDNS auto-discovery"
    print(f"[e2e] connecting to microscope via {target} ...")
    try:
        scope = connect(args.host)
    except AcquisitionError as exc:
        print(f"[e2e] FATAL: could not connect: {exc}", file=sys.stderr)
        return 2

    start = dict(scope.position)
    print(f"[e2e] connected. start position: {start}")
    print(
        f"[e2e] running auto survey: overview {args.overview_rows}x{args.overview_cols} "
        f"@ step ({args.overview_step_x},{args.overview_step_y}), "
        f"focus_map={args.focus_map}, correlate={args.correlate}"
    )

    t0 = time.time()
    survey_ok = False
    survey_err: str | None = None
    try:
        result = run_auto_survey(
            client=scope,
            out_file=out_file,
            overview_rows=args.overview_rows,
            overview_cols=args.overview_cols,
            overview_step_x=args.overview_step_x,
            overview_step_y=args.overview_step_y,
            overlap=args.overlap,
            autofocus=not args.focus_map,  # focus map replaces per-tile autofocus
            correlate=args.correlate,
            min_area_frac=args.min_area_frac,
            focus_map=args.focus_map,
        )
        survey_ok = True
        print(
            f"[e2e] survey finished in {time.time() - t0:.1f}s: "
            f"{result.width}x{result.height} mosaic from {result.tile_count} tiles"
        )
    except (AcquisitionError, SurveyError, FocusError, StitchError) as exc:
        survey_err = f"{type(exc).__name__}: {exc}"
        print(f"[e2e] survey FAILED after {time.time() - t0:.1f}s: {survey_err}", file=sys.stderr)
    finally:
        # Best-effort: leave the stage where we found it.
        try:
            scope.move(start, absolute=True)
            print(f"[e2e] stage returned to start {start}")
        except Exception as exc:
            print(f"[e2e] WARNING: could not return stage to start: {exc}", file=sys.stderr)

    checks: list[Check] = [Check("survey completed", survey_ok, survey_err or "ran to completion")]
    if survey_ok:
        checks.extend(validate(out_file, expect_focus_map=args.focus_map))

    print("\n[e2e] validation report:")
    print(_fmt(checks))
    n_pass = sum(1 for c in checks if c.ok)
    all_ok = all(c.ok for c in checks)
    print(f"\n[e2e] {n_pass}/{len(checks)} checks passed -> {'PASS' if all_ok else 'FAIL'}")
    print(f"[e2e] artifacts in: {args.out_dir.resolve()}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
