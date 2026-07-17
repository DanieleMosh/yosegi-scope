"""Tests for the focus-map module: sampling, plane fit, and prediction.

No microscope: a fake scope returns a Z that follows a known plane so the fit
can be checked exactly.
"""

from __future__ import annotations

import pytest

from yosegi.focus import (
    FocusError,
    FocusMap,
    build_focus_map,
    sample_points_for_region,
)


class _TiltScope:
    """Fake scope whose autofocus lands on a known plane z = a*x + b*y + c.

    Records how many times autofocus ran and the points it was asked to focus at.
    """

    def __init__(self, a: float = 0.0, b: float = 0.0, c: float = 100.0) -> None:
        self._a, self._b, self._c = a, b, c
        self._pos = {"x": 0, "y": 0, "z": 0}
        self.autofocus_calls = 0
        self.focused_at: list[tuple[int, int]] = []

    @property
    def position(self) -> dict[str, int]:
        return dict(self._pos)

    def move(self, position: dict[str, int], absolute: bool = True) -> None:
        if absolute:
            self._pos = {k: int(position[k]) for k in ("x", "y", "z")}
        else:
            for k in ("x", "y", "z"):
                self._pos[k] += int(position[k])

    def autofocus(self, dz: int = 2000) -> None:
        self.autofocus_calls += 1
        x, y = self._pos["x"], self._pos["y"]
        self.focused_at.append((x, y))
        self._pos["z"] = int(round(self._a * x + self._b * y + self._c))


def test_build_focus_map_fits_a_tilted_plane() -> None:
    scope = _TiltScope(a=0.01, b=-0.02, c=500.0)
    points = [(0, 0), (1000, 0), (0, 1000), (1000, 1000)]
    fmap = build_focus_map(scope, points)
    assert isinstance(fmap, FocusMap)
    assert scope.autofocus_calls == 4
    # Predicted Z at a new point matches the underlying plane (within rounding).
    assert fmap.z_at(500, 500) == pytest.approx(0.01 * 500 - 0.02 * 500 + 500, abs=1)
    # z_at clamps to the sampled range, so an extrapolated point can't run away.
    zs = [s[2] for s in fmap.samples]
    assert min(zs) <= fmap.z_at(100000, 100000) <= max(zs)


def test_build_focus_map_constant_when_flat() -> None:
    scope = _TiltScope(a=0.0, b=0.0, c=300.0)
    fmap = build_focus_map(scope, [(0, 0), (100, 0), (0, 100)])
    assert fmap.z_at(9999, -9999) == 300


def test_build_focus_map_falls_back_to_mean_for_few_points() -> None:
    scope = _TiltScope(a=0.05, b=0.0, c=100.0)  # z = 100 at x=0, 150 at x=1000
    fmap = build_focus_map(scope, [(0, 0), (1000, 0)])  # only 2 points -> mean
    assert fmap.coeffs[0] == 0.0 and fmap.coeffs[1] == 0.0
    assert fmap.coeffs[2] == pytest.approx(125.0)  # mean of 100 and 150


def test_build_focus_map_falls_back_for_collinear_points() -> None:
    scope = _TiltScope(a=0.0, b=0.1, c=0.0)  # z varies only in y
    # Three points but all on the line x=0 -> degenerate for a full plane fit.
    fmap = build_focus_map(scope, [(0, 0), (0, 500), (0, 1000)])
    assert fmap.coeffs[0] == 0.0 and fmap.coeffs[1] == 0.0
    # Mean of z at y=0,500,1000 = (0 + 50 + 100) / 3.
    assert fmap.coeffs[2] == pytest.approx(50.0)


def test_build_focus_map_rejects_empty_points() -> None:
    with pytest.raises(FocusError, match="at least one sample"):
        build_focus_map(_TiltScope(), [])


def test_build_focus_map_wraps_autofocus_failure() -> None:
    class _Boom(_TiltScope):
        def autofocus(self, dz: int = 2000) -> None:
            raise RuntimeError("stage jam")

    with pytest.raises(FocusError, match="autofocus failed"):
        build_focus_map(_Boom(), [(0, 0)])


def test_focus_map_z_at_clamps_to_sampled_range() -> None:
    fmap = FocusMap(coeffs=(1.0, 0.0, 0.0), samples=[(0, 0, 0), (100, 0, 100)], z_min=0, z_max=100)
    assert fmap.z_at(-50, 0) == 0     # below range clamps to z_min
    assert fmap.z_at(500, 0) == 100   # above range clamps to z_max
    assert fmap.z_at(50, 0) == 50


def test_sample_points_spread_across_region() -> None:
    pts = sample_points_for_region((0, 0, 1000, 800), max_points=5)
    assert len(pts) == 5
    assert (500, 400) in pts  # centre
    xs = {p[0] for p in pts}
    ys = {p[1] for p in pts}
    assert len(xs) > 1 and len(ys) > 1  # spread in both axes -> plane is fittable


def test_sample_points_zero_size_region_collapses_to_centre() -> None:
    # A degenerate (zero-area) region: centre and all quarter-insets coincide, so
    # de-duplication leaves a single point (no spurious duplicate samples).
    pts = sample_points_for_region((10, 10, 10, 10), max_points=5)
    assert pts == [(10, 10)]


def test_sample_points_respects_max_points() -> None:
    pts = sample_points_for_region((0, 0, 1000, 1000), max_points=2)
    assert len(pts) == 2
