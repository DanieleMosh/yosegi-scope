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
        self.visited: list[tuple[int, int]] = []
        # Identity CSM -> 1 pixel = 1 stage step, no rotation.
        self.csm = [[1.0, 0.0], [0.0, 1.0]]

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
        pass

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


@requires_ofs
def test_run_auto_survey_full_pipeline(tmp_path: Path) -> None:
    out = tmp_path / "mosaic.jpg"
    scope = _Scope(sample_x_range=(-150, 150), sample_y_range=(-150, 150), tile_size=(60, 60))
    result = run_auto_survey(
        client=scope,
        out_file=out,
        overview_rows=5,
        overview_cols=5,
        overview_step_x=80,
        overview_step_y=80,
        overlap=0.2,
        autofocus=False,
        correlate=False,  # stage-only stitch -> deterministic, no correlation flakiness
        min_area_frac=0.005,
    )
    assert out.exists() and result.path == out
    assert result.tile_count > 0
    # Overview side-products land alongside the mosaic for debugging.
    assert (tmp_path / "mosaic_overview.jpg").exists()
    assert (tmp_path / "mosaic_overview").is_dir()
    assert (tmp_path / "mosaic_tiles").is_dir()
    # Overview pass came first (5 * 5 = 25 captures), then the planned scan.
    assert scope.captures > 25


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
