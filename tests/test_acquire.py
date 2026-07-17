"""Tests for the acquisition raster, using an in-memory fake microscope."""

import json
from pathlib import Path

import PIL.Image
import pytest

from yosegi.acquire import (
    AcquisitionError,
    fetch_tiles,
    fetch_tiles_at_positions,
    snake_cells,
)


class FakeMicroscope:
    """In-memory stand-in for MicroscopeClient.

    Tracks XYZ via move/move_rel, returns a tiny PIL image from capture_image,
    and records call order so tests can assert raster behavior.
    """

    def __init__(
        self,
        start: tuple[int, int, int] = (1000, 2000, 3000),
        csm: list[list[float]] | None = None,
    ) -> None:
        self._pos = {"x": start[0], "y": start[1], "z": start[2]}
        self.captures = 0
        self.autofocus_calls = 0
        self.calibrate_calls = 0
        self.csm = csm
        self.move_history: list[dict] = []
        self.visited: list[tuple[int, int]] = []

    @property
    def position(self) -> dict[str, int]:
        return dict(self._pos)

    def pull_settings(self) -> dict:
        ext = {}
        if self.csm is not None:
            ext["org.openflexure.camera_stage_mapping"] = {"image_to_stage_displacement": self.csm}
        return {"extensions": ext}

    def calibrate_xy(self) -> None:
        self.calibrate_calls += 1

    def move(self, position: dict[str, int], absolute: bool = True) -> None:
        if absolute:
            self._pos = {k: int(position[k]) for k in ("x", "y", "z")}
        else:
            for k in ("x", "y", "z"):
                self._pos[k] += int(position[k])
        self.move_history.append({"absolute": absolute, **dict(position)})

    def move_rel(self, position: dict[str, int]) -> None:
        self.move(position, absolute=False)

    def autofocus(self, dz: int = 2000) -> None:
        self.autofocus_calls += 1

    def capture_image(self) -> PIL.Image.Image:
        self.captures += 1
        self.visited.append((self._pos["x"], self._pos["y"]))
        return PIL.Image.new("RGB", (4, 4))


def test_snake_cells_boustrophedon() -> None:
    assert list(snake_cells(2, 3)) == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]


def test_fetch_tiles_captures_grid_and_writes_manifest(tmp_path: Path) -> None:
    scope = FakeMicroscope(start=(1000, 2000, 3000))
    tiles = fetch_tiles(
        host=None, out_dir=tmp_path, rows=2, cols=3, step_x=100, step_y=50, overlap=0.2,
        calibrate=False, client=scope,
    )

    assert len(tiles) == 6
    assert scope.captures == 6
    for r in range(2):
        for c in range(3):
            assert (tmp_path / f"tile_r{r:02d}_c{c:02d}.jpg").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["grid"] == {"rows": 2, "cols": 3}
    assert manifest["step"] == {"x": 100, "y": 50}
    assert manifest["overlap"] == 0.2
    assert len(manifest["tiles"]) == 6
    assert {t["filename"] for t in manifest["tiles"]} == {
        f"tile_r{r:02d}_c{c:02d}.jpg" for r in range(2) for c in range(3)
    }

    # returned to the starting position (last move is an absolute move home)
    assert scope.position == {"x": 1000, "y": 2000, "z": 3000}
    assert scope.move_history[-1]["absolute"] is True


def test_fetch_tiles_stage_coords_follow_snake(tmp_path: Path) -> None:
    scope = FakeMicroscope(start=(0, 0, 0))
    tiles = fetch_tiles(
        host=None, out_dir=tmp_path, rows=2, cols=3, step_x=10, step_y=20, calibrate=False, client=scope
    )

    by_rc = {(t.row, t.col): (t.stage_x, t.stage_y) for t in tiles}
    assert by_rc[(0, 0)] == (0, 0)
    assert by_rc[(0, 2)] == (20, 0)
    assert by_rc[(1, 2)] == (20, 20)
    assert by_rc[(1, 0)] == (0, 20)
    assert scope.visited == [(0, 0), (10, 0), (20, 0), (20, 20), (10, 20), (0, 20)]


def test_autofocus_called_only_when_enabled(tmp_path: Path) -> None:
    off = FakeMicroscope()
    fetch_tiles(
        host=None, out_dir=tmp_path / "a", rows=2, cols=2, step_x=1, step_y=1,
        autofocus=False, calibrate=False, client=off,
    )
    assert off.autofocus_calls == 0

    on = FakeMicroscope()
    fetch_tiles(
        host=None, out_dir=tmp_path / "b", rows=2, cols=2, step_x=1, step_y=1,
        autofocus=True, calibrate=False, client=on,
    )
    assert on.autofocus_calls == 4


def test_invalid_grid_raises(tmp_path: Path) -> None:
    scope = FakeMicroscope()
    with pytest.raises(AcquisitionError):
        fetch_tiles(host=None, out_dir=tmp_path / "x", rows=0, cols=3, step_x=1, step_y=1, client=scope)
    # validation happens before any side effects
    assert not (tmp_path / "x").exists()


_CSM = [[0.01, -4.4], [-4.37, 0.0]]


def _read_tile_exif(path: Path) -> dict:
    import piexif

    exif = piexif.load(str(path))
    raw = exif["Exif"][piexif.ExifIFD.UserComment]
    return json.loads(raw.decode())  # raw UTF-8 JSON, matching openflexure-stitching


def test_csm_recorded_in_manifest_and_exif(tmp_path: Path) -> None:
    scope = FakeMicroscope(start=(3600, -5600, 0), csm=_CSM)
    tiles = fetch_tiles(
        host=None, out_dir=tmp_path, rows=2, cols=2, step_x=400, step_y=400, client=scope
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["camera_stage_mapping"] == _CSM

    uc = _read_tile_exif(tiles[0].path)
    assert uc["stage"]["position"]["x"] == 3600
    assert uc["camera_stage_mapping"]["image_to_stage_displacement_matrix"] == _CSM


def test_missing_csm_triggers_calibration_then_records_none(tmp_path: Path) -> None:
    # no stored CSM and calibrate_xy is a no-op fake -> calibration attempted, CSM stays None
    scope = FakeMicroscope(csm=None)
    fetch_tiles(
        host=None, out_dir=tmp_path, rows=2, cols=2, step_x=400, step_y=400,
        calibrate=True, client=scope,
    )
    assert scope.calibrate_calls == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["camera_stage_mapping"] is None


def test_no_calibration_when_disabled(tmp_path: Path) -> None:
    scope = FakeMicroscope(csm=None)
    fetch_tiles(
        host=None, out_dir=tmp_path, rows=2, cols=2, step_x=400, step_y=400,
        calibrate=False, client=scope,
    )
    assert scope.calibrate_calls == 0


# --- fetch_tiles_at_positions ----------------------------------------------


def test_positions_scan_captures_at_the_planned_positions(tmp_path: Path) -> None:
    """Regression: the first tile must be captured AT positions[0], not at the
    scope's current location.

    _capture_at_plan skips the move before its first entry (right for
    fetch_tiles, where plan[0] == current position). For an arbitrary planned
    scan the scope may be parked far away (e.g. the overview start), so without
    an explicit move to positions[0] every tile is offset and the scan images
    the wrong region. Here the scope starts at (1000, 2000) but the plan begins
    at (5000, 6000): the captured positions must equal the plan exactly.
    """
    scope = FakeMicroscope(start=(1000, 2000, 3000))
    positions = [(5000, 6000), (5100, 6000), (5100, 6100), (5000, 6100)]
    tiles = fetch_tiles_at_positions(
        client=scope, out_dir=tmp_path, positions=positions,
        rows=2, cols=2, autofocus=False,
    )
    assert scope.visited == positions
    # (row, col) are derived from snake order over the 2x2 grid.
    by_rc = {(t.row, t.col): (t.stage_x, t.stage_y) for t in tiles}
    assert by_rc[(0, 0)] == (5000, 6000)
    assert by_rc[(0, 1)] == (5100, 6000)
    assert by_rc[(1, 1)] == (5100, 6100)
    assert by_rc[(1, 0)] == (5000, 6100)
    # Scope returned home afterwards.
    assert scope.position == {"x": 1000, "y": 2000, "z": 3000}


def test_positions_scan_writes_manifest(tmp_path: Path) -> None:
    scope = FakeMicroscope(start=(0, 0, 0), csm=_CSM)
    positions = [(10, 20), (30, 20)]
    fetch_tiles_at_positions(
        client=scope, out_dir=tmp_path, positions=positions,
        rows=1, cols=2, autofocus=False,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema"] == "yosegi.acquire/1"
    assert manifest["grid"] == {"rows": 1, "cols": 2}
    assert manifest["planned_positions"] == [[10, 20], [30, 20]]
    assert manifest["camera_stage_mapping"] == _CSM
    assert len(manifest["tiles"]) == 2


def test_positions_scan_rejects_length_mismatch(tmp_path: Path) -> None:
    scope = FakeMicroscope()
    with pytest.raises(AcquisitionError, match="positions has"):
        fetch_tiles_at_positions(
            client=scope, out_dir=tmp_path, positions=[(0, 0)], rows=2, cols=2,
        )


def test_positions_scan_focus_z_sets_z_and_skips_autofocus(tmp_path: Path) -> None:
    """With focus_z the scope moves Z per tile (no per-tile autofocus), and each
    tile records the focus-map Z in EXIF/Tile."""
    scope = FakeMicroscope(start=(0, 0, 0))
    positions = [(0, 0), (100, 0), (100, 100), (0, 100)]
    focus_z = [500, 510, 520, 530]
    tiles = fetch_tiles_at_positions(
        client=scope, out_dir=tmp_path, positions=positions,
        rows=2, cols=2, focus_z=focus_z, autofocus=True,  # autofocus ignored when focus_z given
    )
    # focus_z takes precedence over autofocus -> no autofocus calls.
    assert scope.autofocus_calls == 0
    by_pos = {(t.stage_x, t.stage_y): t.stage_z for t in tiles}
    assert by_pos[(0, 0)] == 500
    assert by_pos[(100, 0)] == 510
    assert by_pos[(100, 100)] == 520
    assert by_pos[(0, 100)] == 530


def test_positions_scan_rejects_focus_z_length_mismatch(tmp_path: Path) -> None:
    scope = FakeMicroscope()
    with pytest.raises(AcquisitionError, match="focus_z has"):
        fetch_tiles_at_positions(
            client=scope, out_dir=tmp_path, positions=[(0, 0), (10, 0)],
            rows=1, cols=2, focus_z=[100],
        )
