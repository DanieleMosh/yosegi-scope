"""Build a per-region focus surface so tiles are focused without re-autofocusing.

Autofocusing at every tile is slow and, on a tilted slide, noisy. Automated slide
scanners instead measure focus at a few points, fit a smooth ``Z(x, y)`` surface
(a "focus map"), and set each tile's Z from that surface. This module does the
fit: :func:`build_focus_map` samples autofocus at a handful of in-tissue stage
points, fits a plane (or falls back to a constant), and returns a
:class:`FocusMap` that predicts Z at any ``(x, y)``.

The fit is a least-squares plane ``z = a*x + b*y + c`` -- the right model for a
slide that is flat but tilted on the stage. With fewer than three
non-degenerate samples there is no plane to fit, so the mean sampled Z is used
everywhere. Heavy imports (numpy) stay lazy; failures normalise into
:class:`FocusError` to match the other ``*Error`` types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class FocusError(RuntimeError):
    """Raised when a focus surface cannot be sampled or fitted."""


class FocusScope(Protocol):
    """The bits of the microscope client needed to sample focus.

    A structural subset of :class:`yosegi.acquire.Microscope`, so the real client
    and the test doubles satisfy it without a new adapter.
    """

    @property
    def position(self) -> dict[str, int]: ...
    def move(self, position: dict[str, int], absolute: bool = True) -> object: ...
    def autofocus(self, dz: int = 2000) -> object: ...


@dataclass(frozen=True)
class FocusMap:
    """A fitted focus surface ``Z(x, y)`` over stage coordinates (in steps).

    ``coeffs`` are ``(a, b, c)`` of the plane ``z = a*x + b*y + c``; a constant
    fallback is ``(0, 0, mean_z)``. ``samples`` are the ``(x, y, z)`` autofocus
    measurements the fit came from, kept for logging/provenance. ``z_min``/
    ``z_max`` bound predictions so an extrapolated plane can't drive the stage to
    an absurd Z on a point far outside the sampled span.
    """

    coeffs: tuple[float, float, float]
    samples: list[tuple[int, int, int]]
    z_min: int
    z_max: int

    def z_at(self, x: float, y: float) -> int:
        """Predicted focus Z at stage ``(x, y)``, clamped to the sampled range."""
        a, b, c = self.coeffs
        z = a * x + b * y + c
        return int(round(min(self.z_max, max(self.z_min, z))))


def _fit_plane(samples: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    """Least-squares plane ``z = a*x + b*y + c`` through ``samples``.

    Falls back to ``(0, 0, mean_z)`` when there are fewer than three samples or
    the ``(x, y)`` points are degenerate (collinear / coincident), i.e. whenever
    the normal equations are rank-deficient and a unique plane does not exist.
    """
    import numpy as np

    zs = np.array([s[2] for s in samples], dtype=float)
    mean_z = float(zs.mean())
    if len(samples) < 3:
        return (0.0, 0.0, mean_z)

    a_mat = np.array([[s[0], s[1], 1.0] for s in samples], dtype=float)
    # Rank-deficient design (collinear xy) -> no unique plane; use the mean.
    if np.linalg.matrix_rank(a_mat[:, :2] - a_mat[:, :2].mean(axis=0)) < 2:
        return (0.0, 0.0, mean_z)
    coeffs, *_ = np.linalg.lstsq(a_mat, zs, rcond=None)
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))


def build_focus_map(
    scope: FocusScope,
    points: list[tuple[int, int]],
    *,
    autofocus_dz: int = 2000,
) -> FocusMap:
    """Autofocus at each ``(x, y)`` in ``points`` and fit a focus surface.

    Moves the stage to each sample point (absolute XY, keeping the current Z),
    runs ``autofocus``, and records the resulting stage Z. Fits a least-squares
    plane through the samples (or a constant when there are too few / degenerate
    points) and returns a :class:`FocusMap`. The stage is left at the last sample
    point; callers that need the original position should restore it.

    Raises :class:`FocusError` if ``points`` is empty or no sample yields a Z.
    """
    if not points:
        raise FocusError("focus map needs at least one sample point")

    start = dict(scope.position)
    z0 = int(start.get("z", 0))
    samples: list[tuple[int, int, int]] = []
    for x, y in points:
        try:
            scope.move({"x": int(x), "y": int(y), "z": z0}, absolute=True)
            scope.autofocus(autofocus_dz)
            z = int(dict(scope.position).get("z", z0))
        except Exception as exc:  # network/hardware hiccup on one point
            raise FocusError(f"autofocus failed at ({x}, {y}): {exc}") from exc
        samples.append((int(x), int(y), z))

    if not samples:
        raise FocusError("no focus samples were collected")

    coeffs = _fit_plane(samples)
    zs = [s[2] for s in samples]
    return FocusMap(coeffs=coeffs, samples=samples, z_min=min(zs), z_max=max(zs))


def sample_points_for_region(
    bbox: tuple[int, int, int, int],
    *,
    max_points: int = 5,
) -> list[tuple[int, int]]:
    """Pick up to ``max_points`` focus-sample stage positions inside a region bbox.

    ``bbox`` is ``(x0, y0, x1, y1)`` in stage steps. Returns the centre plus, when
    room and budget allow, points toward the corners so the plane fit sees spread
    in both axes (a centre-only sample would only pin the constant term). Points
    are de-duplicated, so a tiny region collapses to just its centre.
    """
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    # Inset the corners a little so samples stay off the very edge of the tissue.
    ix0, iy0 = x0 + (x1 - x0) // 4, y0 + (y1 - y0) // 4
    ix1, iy1 = x1 - (x1 - x0) // 4, y1 - (y1 - y0) // 4
    candidates = [(cx, cy), (ix0, iy0), (ix1, iy1), (ix0, iy1), (ix1, iy0)]

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= max(1, max_points):
            break
    return out


__all__ = [
    "FocusError",
    "FocusMap",
    "FocusScope",
    "build_focus_map",
    "sample_points_for_region",
]
