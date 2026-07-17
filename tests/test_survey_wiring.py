"""Tests for the run_auto_survey orchestrator.

Wires the existing ``FakeMicroscope`` pattern to ``run_auto_survey``: a fake
scope returns synthetic frames, and we assert the overview pass + planned scan
behaviour without touching real hardware. The full end-to-end test that
exercises the final ``stitch_tiles`` call requires libvips and is skipped when
``openflexure_stitching`` cannot be imported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from yosegi.survey import SurveyError, run_auto_survey

# Skip libvips-dependent tests if openflexure-stitching can't import.
try:
    import openflexure_stitching  # noqa: F401

    _OFS_AVAILABLE = True
except Exception:
    _OFS_AVAILABLE = False

requires_ofs = pytest.mark.skipif(
    not _OFS_AVAILABLE, reason="requires libvips / openflexure-stitching"
)


class _Scope:
    """Identity-CSM fake scope that returns dark frames near the origin only.

    The ``capture_image`` function fakes a sample concentrated within a stage
    bounding box (``sample_x_range``, ``sample_y_range``). Frames captured
    inside that bbox are dark and textured (tissue); frames outside are bright
    and flat (empty slide). This drives ``detect_sample_bbox`` to a known bbox
    on the stitched overview without needing a real microscope.
    """

    def __init__(
        self,
        *,
        sample_x_range: tuple[int, int] = (-300, 300),
        sample_y_range: tuple[int, int] = (-300, 300),
        tile_size: tuple[int, int] = (60, 60),
    ) -> None:
        self._pos = {"x": 0, "y": 0, "z": 0}
        self._tile_w, self._tile_h = tile_size
        self.sample_x_range = sample_x_range
        self.sample_y_range = sample_y_range
        self.captures = 0
        self.autofocus_calls = 0
        self.visited: list[tuple[int, int]] = []
        # Identity CSM -> 1 pixel = 1 stage step, no rotation.
        self.csm = [[1.0, 0.0], [0.0, 1.0]]
        # Focus plane z = 0.01*x + 0.02*y + 200; autofocus lands the stage on it.
        self._focus = (0.01, 0.02, 200.0)

    @property
    def position(self) -> dict[str, int]:
        return dict(self._pos)

    def pull_settings(self) -> dict:
        return {
            "extensions": {
                "org.openflexure.camera_stage_mapping": {
                    "image_to_stage_displacement": self.csm
                }
            }
        }

    def calibrate_xy(self) -> None:  # pragma: no cover (CSM is pre-populated)
        pass

    def move(self, position: dict[str, int], absolute: bool = True) -> None:
        if absolute:
            self._pos = {k: int(position[k]) for k in ("x", "y", "z")}
        else:
            for k in ("x", "y", "z"):
                self._pos[k] += int(position[k])

    def move_rel(self, position: dict[str, int]) -> None:
        self.move(position, absolute=False)

    def autofocus(self, dz: int = 2000) -> None:
        self.autofocus_calls += 1
        a, b, c = self._focus
        self._pos["z"] = int(round(a * self._pos["x"] + b * self._pos["y"] + c))

    def capture_image(self) -> PIL.Image.Image:
        self.captures += 1
        x, y = self._pos["x"], self._pos["y"]
        self.visited.append((x, y))
        in_x = self.sample_x_range[0] <= x <= self.sample_x_range[1]
        in_y = self.sample_y_range[0] <= y <= self.sample_y_range[1]
        if in_x and in_y:
            # Dark, textured frame -> survives both intensity + variance cues.
            rng = np.random.default_rng(seed=(abs(x) * 17 + abs(y) * 11) & 0xFFFF)
            arr = rng.integers(20, 80, size=(self._tile_h, self._tile_w, 3), dtype=np.uint8)
        else:
            # Bright, flat frame -> reads as empty slide.
            arr = np.full((self._tile_h, self._tile_w, 3), 245, dtype=np.uint8)
        return PIL.Image.fromarray(arr)


class _TwoRegionScope(_Scope):
    """Fake scope with two separated tissue blocks and empty slide between them.

    Frames are dark/textured (tissue) inside either of two stage boxes and bright
    (empty) everywhere else. Drives ``run_auto_survey`` to plan a *sparse* scan
    that skips the gap between the regions.
    """

    def __init__(self, tile_size: tuple[int, int] = (60, 60)) -> None:
        super().__init__(tile_size=tile_size)
        # The overview raster starts at the scope origin (0,0) and steps in the
        # +x/+y direction, so tissue must sit in the positive quadrant it covers
        # (here [0, 320] on each axis for a 9x9 grid at step 40). Two blocks with
        # a wide empty gap between them (x 100..220) stay two components through
        # the morphological close.
        self._boxes = [
            ((0, 90), (0, 320)),      # (x_range, y_range) left block
            ((230, 320), (0, 320)),   # right block
        ]

    def capture_image(self) -> PIL.Image.Image:
        self.captures += 1
        x, y = self._pos["x"], self._pos["y"]
        self.visited.append((x, y))
        in_tissue = any(
            xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1] for xr, yr in self._boxes
        )
        if in_tissue:
            rng = np.random.default_rng(seed=(abs(x) * 17 + abs(y) * 11) & 0xFFFF)
            arr = rng.integers(20, 80, size=(self._tile_h, self._tile_w, 3), dtype=np.uint8)
        else:
            arr = np.full((self._tile_h, self._tile_w, 3), 245, dtype=np.uint8)
        return PIL.Image.fromarray(arr)


@requires_ofs
def test_run_auto_survey_skips_empty_tiles_between_regions(tmp_path: Path) -> None:
    """End-to-end: two separated tissue blocks -> the high-res scan skips the
    empty gap, so it captures fewer tiles than a dense grid over the union bbox."""
    out = tmp_path / "mosaic.jpg"
    scope = _TwoRegionScope(tile_size=(60, 60))
    result = run_auto_survey(
        client=scope,
        out_file=out,
        overview_rows=9,
        overview_cols=9,
        overview_step_x=40,
        overview_step_y=40,
        overlap=0.2,
        autofocus=False,
        correlate=False,
        min_area_frac=0.002,
    )
    assert out.exists()
    overview_captures = 9 * 9
    highres_tiles = scope.captures - overview_captures
    assert highres_tiles > 0
    # The stitched mosaic used every high-res tile captured.
    assert result.tile_count == highres_tiles

    # Reconstruct the dense grid the planner would have produced over the same
    # union bbox and confirm the tissue gate dropped tiles from the empty gap.
    import json

    from yosegi.survey import ScanPlan, detect_sample_regions, plan_survey, plan_tile_grid

    manifest = json.loads((tmp_path / "mosaic_overview" / "manifest.json").read_text())
    csm = manifest["camera_stage_mapping"]
    from PIL import Image

    with Image.open(tmp_path / "mosaic_overview.jpg") as ov:
        tissue = detect_sample_regions(np.asarray(ov.convert("RGB")), min_area_frac=0.002)
    # Two distinct regions were detected.
    assert len(tissue.regions) >= 2
    gated: ScanPlan = plan_survey(
        tissue, overview_origin_stage=(0, 0), overview_csm=csm,
        tile_size_px=(60, 60), overlap=0.2,
    )
    dense: ScanPlan = plan_tile_grid(
        tissue.bbox, overview_origin_stage=(0, 0), overview_csm=csm,
        tile_size_px=(60, 60), overlap=0.2,
    )
    assert gated.tile_count < dense.tile_count


@requires_ofs
def test_run_auto_survey_focus_map_replaces_per_tile_autofocus(tmp_path: Path) -> None:
    """With focus_map=True the scope autofocuses only at the focus-sample points,
    not once per high-res tile, and each tile's Z follows the fitted plane."""
    out = tmp_path / "mosaic.jpg"
    scope = _Scope(sample_x_range=(0, 320), sample_y_range=(0, 320), tile_size=(60, 60))
    result = run_auto_survey(
        client=scope,
        out_file=out,
        overview_rows=9,
        overview_cols=9,
        overview_step_x=40,
        overview_step_y=40,
        overlap=0.2,
        autofocus=False,
        autofocus_once=False,
        correlate=False,
        min_area_frac=0.002,
        focus_map=True,
        focus_points_per_region=5,
    )
    assert out.exists()
    highres_tiles = result.tile_count
    # Autofocus ran only while sampling the focus map (a small, bounded number of
    # points), NOT once per high-res tile.
    assert 0 < scope.autofocus_calls <= 5  # one region, up to 5 sample points
    assert scope.autofocus_calls < highres_tiles

    # Each captured high-res tile carries a Z from the fitted plane
    # (z = 0.01*x + 0.02*y + 200), i.e. clearly non-zero and stage-dependent.
    import json

    manifest = json.loads((tmp_path / "mosaic_tiles" / "manifest.json").read_text())
    assert manifest["focus_map"] is True and manifest["autofocus"] is False
    zs = [t["stage_z"] for t in manifest["tiles"]]
    assert all(z is not None and z > 100 for z in zs)
    assert len(set(zs)) > 1  # the plane tilts, so tile Z varies across the scan


@requires_ofs
def test_run_auto_survey_auto_expand_encloses_a_clipped_sample(tmp_path: Path) -> None:
    """A sample larger than the initial overview must trigger overview growth.

    The sample spans a stage box far wider than a 3x3 overview at step 40 covers,
    so the first overview clips it (tissue touches every edge). With auto_expand
    the overview grows outward and re-detects until the tissue is enclosed -- so
    the final overview canvas and total overview captures both exceed the initial
    3x3 grid, and the sample is no longer edge-clipped.
    """
    from PIL import Image

    from yosegi.survey import bbox_touches_edges, detect_sample_regions

    out = tmp_path / "mosaic.jpg"
    # Sample much larger than a 3x3@40 overview (which covers ~120x120 steps).
    scope = _Scope(sample_x_range=(-30, 500), sample_y_range=(-30, 500), tile_size=(60, 60))
    run_auto_survey(
        client=scope,
        out_file=out,
        overview_rows=3,
        overview_cols=3,
        overview_step_x=40,
        overview_step_y=40,
        overlap=0.2,
        autofocus=False,
        correlate=False,
        min_area_frac=0.002,
        auto_expand=True,
        max_expansions=6,
        expand_increment=2,
    )
    # The overview grew: total overview captures exceed the initial 3*3 = 9.
    # (Each expansion re-rasters a larger grid, so captures accumulate well past 9.)
    assert scope.captures > 9
    # The final overview no longer clips the sample on all sides.
    with Image.open(tmp_path / "mosaic_overview.jpg") as ov:
        arr = np.asarray(ov.convert("RGB"))
    tissue = detect_sample_regions(arr, min_area_frac=0.002)
    touch = bbox_touches_edges(tissue.bbox, arr.shape[:2])
    assert not touch.any, f"sample still clipped after auto-expand: {touch}"


@requires_ofs
def test_run_auto_survey_full_pipeline(tmp_path: Path) -> None:
    out = tmp_path / "mosaic.jpg"
    # Make the sample large enough that the detected bbox is at least 2x2 tiles,
    # so plan_tile_grid yields several tiles for the final stitch.
    # Overview step < tile size so synthetic tile blobs join into one component.
    scope = _Scope(sample_x_range=(-300, 300), sample_y_range=(-300, 300), tile_size=(60, 60))
    result = run_auto_survey(
        client=scope,
        out_file=out,
        overview_rows=7,
        overview_cols=7,
        overview_step_x=40,
        overview_step_y=40,
        overlap=0.2,
        autofocus=False,
        correlate=False,  # stage-only stitch -> deterministic, no correlation flakiness
        min_area_frac=0.005,
    )
    assert out.exists() and result.path == out
    assert result.tile_count > 1  # planned grid must be more than 1 tile
    # Overview side-products land alongside the mosaic for debugging.
    assert (tmp_path / "mosaic_overview.jpg").exists()
    assert (tmp_path / "mosaic_overview").is_dir()
    assert (tmp_path / "mosaic_tiles").is_dir()
    # Overview pass came first (7 * 7 = 49 captures), then the planned scan.
    assert scope.captures > 49


def test_run_auto_survey_raises_when_overview_has_no_sample(tmp_path: Path) -> None:
    out = tmp_path / "mosaic.jpg"
    # Sample range outside any stage position the overview will visit -> every
    # overview frame reads "empty slide". Use full-overlap stepping (step <
    # tile_size) so the stitched canvas has no black gaps that could be
    # mistaken for tissue. Tile size = 60, step = 30 -> every pixel is covered.
    scope = _Scope(
        sample_x_range=(100_000, 100_001),
        sample_y_range=(100_000, 100_001),
        tile_size=(60, 60),
    )
    with pytest.raises(SurveyError, match="no sample detected"):
        run_auto_survey(
            client=scope,
            out_file=out,
            overview_rows=4,
            overview_cols=4,
            overview_step_x=30,
            overview_step_y=30,
            autofocus=False,
            correlate=False,
        )
    # The overview pass still ran (4*4 = 16 captures); the failure is mid-pipeline.
    assert scope.captures == 16
    # The final mosaic must not be written when survey fails.
    assert not out.exists()


def test_run_auto_survey_calls_overview_then_plan(tmp_path: Path) -> None:
    """The orchestrator must run a complete overview pass before the planned scan."""
    out = tmp_path / "mosaic.jpg"
    scope = _Scope(sample_x_range=(-150, 150), sample_y_range=(-150, 150))
    try:
        run_auto_survey(
            client=scope,
            out_file=out,
            overview_rows=3,
            overview_cols=3,
            overview_step_x=100,
            overview_step_y=100,
            autofocus=False,
            correlate=False,
        )
    except Exception:
        # Final stitch may fail without libvips -- that's fine for this assertion.
        pass
    # The overview snake visits 9 distinct (x, y) cells starting at the origin.
    overview_visits = scope.visited[:9]
    assert (0, 0) in overview_visits
    assert (100, 0) in overview_visits
    assert (200, 0) in overview_visits
    # After the 9 overview captures, additional captures from the planned scan happen.
    assert len(scope.visited) > 9
